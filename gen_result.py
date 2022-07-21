import torch
import numpy as np

import json
from model01 import HandModel 
from dme_data import DMEDataset
from torch.utils.data import DataLoader
import cv2

def run(model_path, result_folder, test_json_path, model_channel=1):
    model = HandModel(model_channel).cuda()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    model.eval()
    
    # load test set
    with open(test_json_path, 'r') as f:
        data = json.load(f)
    
    test_set = DMEDataset(data)
    test_set_loader = DataLoader(test_set)

    for iteration, dat in enumerate(test_set_loader):

        iteration += 1
        image = dat['image'].cuda()/255
        image = image.unsqueeze(1)
          
        output = model(image)
        pred_l = output[1]

        pred_s = output[3]
        
        pred_l = pred_l[0].mean(0).T
        pred_l = np.array(pred_l.cpu().detach().numpy())
        pred_s = pred_s[0].max(0)[0].T
        pred_s = np.array(pred_s.cpu().detach().numpy())
        
        cv2.imwrite(result_folder + 'pred_l_%d.jpg'%iteration,pred_l*255)
        cv2.imwrite(result_folder + 'pred_s_%d.jpg'%iteration, pred_s*255)

        # gen html
        html_content = '''
        <html>
        <body>
        <img src='%s'>
        <img src='%s'>
        <img src='%s'>
        </body>
        </html>
        '''%(
           '../'+ str(dat['image_path']),
           './pred_l_%d.jpg'%iteration,
           './pred_s_%d.jpg'%iteration,
             )
        with open(result_folder + 'result_%s.html'%iteration, 'w') as f:
            f.write(html_content)

            
        print(iteration, len(test_set_loader))

if __name__ == "__main__":
    model_path = 'save/train01.pyepoch0000000078.model'
    result_path = 'temp_result_01/'
    test_json_path = 'validation_json_with_covered'
    run(model_path, result_path, test_json_path, model_channel=1)














    
