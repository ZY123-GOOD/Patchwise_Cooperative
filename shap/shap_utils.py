import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from typing import List
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import copy
import numpy as np

SYSTEM_MSG = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
IGNORE_INDEX = -100

def format_question(question, conversation_style='chat'):
    if conversation_style == 'plain': # for 1st stage model
        question = DEFAULT_IMAGE_TOKEN + question
    elif conversation_style == 'chat': # for 2nd stage model
        question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    else:
        raise NotImplementedError()
    return question

def format_answer(answer, conversation_style='chat'):
    if conversation_style == 'plain': # for 1st stage model
        answer = answer + "\n"
    elif conversation_style == 'chat': # for 2nd stage model
        answer = answer + "</s>"
    else:
        raise NotImplementedError()
    return answer


def eval_model(tokenizer, model, images, texts,question_template, answer_template):

    with torch.inference_mode():
        assert len(images) == len(texts), "Number of images and texts must match"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        questions = [SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: " for question in questions]
        
        answers = [format_answer(answer, conversation_style='chat') for answer in answers]
        
        prompts = [qs + ans for qs, ans in zip(questions, answers)]
        
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in prompts]
        labels = copy.deepcopy(input_ids)
        for label, qs in zip(labels, questions):
            tokenized_len = len(tokenizer_image_token(qs, tokenizer))
            if qs[-1] == " ":
                tokenized_len -= 1 # because white space
            label[:tokenized_len] = IGNORE_INDEX

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :tokenizer.model_max_length]
        labels = labels[:, :tokenizer.model_max_length]

        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        
        input_ids, attention_mask, labels = input_ids.to(device='cuda', non_blocking=True), attention_mask.to(device='cuda'), labels.to(dtype=torch.int64, device='cuda', non_blocking=True)
        input_ids, _, attention_mask, past_key_values, inputs_embeds, labels = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            attention_mask,
            None,
            labels,
            images
        )
            

        assert input_ids is None, "input_ids should be None for LLaVA-1.5"
        assert past_key_values is None, "past_key_values should be None for LLaVA-1.5"
        model_input_kwargs = {
            'input_ids': input_ids, # None for LLaVA-1.5
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': inputs_embeds,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': False,
        }
        
        outputs = model.model(
            **model_input_kwargs
        )

        hidden_states = outputs[0]
        logits = model.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        shift_labels = shift_labels.to(shift_logits.device)
        lm_prob = torch.zeros(shift_logits.shape[0])
        
        # print(shift_labels[0].shape,shift_logits[0].shape)
        # break
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(shift_logits[k].to(torch.bfloat16), shift_labels[k].to(torch.int64))).exp()
        
    return lm_prob



### Explanations for a specific layer rather than the last layer.
### You can set the layer_idx in a range of [1,31] for LLaVA-V1.5-7B
def eval_model_specific(tokenizer, model, images, texts,question_template,answer,layer_idx=31):

    with torch.inference_mode():
        questions = [question_template.format(text) for text in texts]
        
        questions = [SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: " for question in questions]
        # print(questions)
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in questions]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)

        input_ids = input_ids[:, :tokenizer.model_max_length]
        
        layer_outputs = []
        def hook(module, input, output):
            layer_outputs.append(output)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)

        generation_output = model.generate(
            input_ids,
            images=images,
            image_sizes=336,
            do_sample= False,
            top_p=None,
            num_beams=1,
            max_new_tokens=35,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        handle.remove()
        
        lm_prob = torch.zeros(len(generation_output.scores[0]))
        for k in range(lm_prob.shape[0]):
            hidden_states = model.model.norm(layer_outputs[0][0])
            logits = model.lm_head(hidden_states)
            logits = torch.softmax(logits, dim=-1)
            probs = logits[k, -1, :]
            # probs = torch.softmax(probs, dim=-1)

            lm_prob[k] = probs[tokenizer(answer).input_ids[1]] + probs[tokenizer(answer.lower()).input_ids[1]]

    return lm_prob




class NeuralNet(nn.Module): ### a simple NN network
    def __init__(self, in_size,bs,lr):
        super(NeuralNet, self).__init__()
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(in_size,int(in_size*2)),nn.CELU(),nn.Linear(int(in_size*2),int(1)),torch.nn.Sigmoid())
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.loss = torch.nn.MSELoss()
        self.bs = bs


    def change_lr(self,lr):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def forward(self,x):
        return self.model(x)
    
    def step(self, x,y):
        self.optimizer.zero_grad()
        loss = self.loss
        output = loss(x, y)
        output.backward()
        self.optimizer.step()
        return output.detach().cpu().numpy()
    
    def step_val(self, x,y):
        self.optimizer.zero_grad()
        loss = self.loss
        output = loss(x, y)
        return output.detach().cpu().numpy()


def learning_feat(target_model,tokenizer,default_question_template,default_answer_template,concept_mask,target_img,target_label,batch,version,layeridx):
        
    batch_img = []
    for i in range(concept_mask.shape[0]):
        input_tensor = transforms.ToTensor()(concept_mask[i])*transforms.ToTensor()(target_img)
        # print(input_tensor.shape)
        batch_img.append((input_tensor.cpu()))
        # if i%200==0:
        #     print("process:",i)
    tmp_dl = DataLoader(dataset = batch_img, batch_size=batch, shuffle =False)
    # print("tmp_dl:",len(tmp_dl))
    
    output = None
    fc_res = None
    idx = 0
    for x in tmp_dl:
        idx += 1
        with torch.no_grad():
            
            if fc_res == None:
                if version==0:
                    output = eval_model(tokenizer, target_model, x, x.shape[0]*[target_label], question_template=default_question_template, answer_template=default_answer_template)
                elif version==1:
                    output = eval_model_specific(tokenizer, target_model, x, x.shape[0]*[target_label], question_template=default_question_template,answer=default_answer_template,layer_idx=layeridx)
                fc_res = output
                # print("fc_res",fc_res)
            else:
                if version==0:
                    output = eval_model(tokenizer, target_model, x, x.shape[0]*[target_label], question_template=default_question_template, answer_template=default_answer_template)
                elif version==1:
                    output = eval_model_specific(tokenizer, target_model, x, x.shape[0]*[target_label], question_template=default_question_template,answer=default_answer_template,layer_idx=layeridx)

                fc_res = torch.cat((fc_res,output))
            del x
            torch.cuda.empty_cache()
    # print("learning_feat finish")
    return fc_res


    
def learn_PIE(target_model,tokenizer,default_question_template,default_answer_template,concept_mask,target_img,target_label,lr,epochs,batch,simu_num,version,layeridx):
    
    masks_tmp = concept_mask.copy()  # (41,224,224)
    
    num_feat = masks_tmp.shape[0]

    bin_x1 = np.random.binomial(1,0.5,size=(simu_num,num_feat)).astype(bool) #### generate samples to learn PIE by default 
    new_mask1 = np.array([masks_tmp[i].sum(0) for i in bin_x1]).astype(bool)   
    
    bin_x2 = np.random.binomial(1,0.5,size=(simu_num,num_feat)).astype(bool) #### generate samples to learn PIE by default
    new_mask2 = np.array([masks_tmp[i].sum(0) for i in bin_x2]).astype(bool)   
    
    bin_x = np.vstack((bin_x1,bin_x2))
    new_mask = np.vstack((new_mask1,new_mask2))
    del bin_x1,bin_x2,new_mask1,new_mask2
    
    probs = learning_feat(target_model,tokenizer,default_question_template,default_answer_template,new_mask,target_img,target_label,batch,version,layeridx)
    
    probs = probs.detach().clone().cpu()
    bin_x_torch = torch.tensor(bin_x.tolist(),dtype=torch.float)
    data = [[x,y] for x,y in zip(bin_x_torch,probs)]
    bs = 300
    
    
    net = NeuralNet(num_feat,bs,lr).cuda()
    
    net.change_lr(lr)
    
    
    data_comb_train = DataLoader(dataset = data[num_feat:], batch_size = bs, shuffle =True)
    
    ##### learning combin
    for epoch in range(epochs):
        loss = 0
        for x,y in data_comb_train:
            pred = net(x.cuda())
            # print("pred:",pred)
            # print("y:",y)
            tmploss = net.step(pred.squeeze(1),y.cuda())
            loss += tmploss*x.shape[0]
    # print("loss:", loss)
        
    net.eval()
    return net
