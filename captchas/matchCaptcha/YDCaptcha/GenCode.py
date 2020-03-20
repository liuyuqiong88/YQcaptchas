
import cv2 
import numpy as np


class YDCap:

    def __init__(self):

        self.model = cv2.imread("captchas/YDCaptcha/yd.yq")
        self.model = cv2.cvtColor(self.model,cv2.COLOR_BGR2GRAY)
        _ , self.model = cv2.threshold(self.model,127,255,cv2.THRESH_BINARY)

    def match_code(self,x):

        code_list = {
        "0" : (10 , 30) ,
        "1" : (30 , 50) ,
        "2" : (50 , 70) ,
        "3" : (70 , 90) ,
        "4" : (90 , 110) ,
        "5" : (110 , 130) ,
        "6" : (130 , 150) ,
        "7" : (150 , 170) ,
        "8" : (170 , 190) ,
        "9" : (190 , 210) ,
        "+" : (210 , 230) ,
        "*" : (230 , 250) ,
        }
        for code,max_min in code_list.items():
            if x >= max_min[0] and x < max_min[1] :
                return code

    def get_code(self,split_img):
        res  = cv2.matchTemplate(self.model,split_img,cv2.TM_SQDIFF_NORMED)
        _,_,min_loc,_ = cv2.minMaxLoc(res)
        return self.match_code(min_loc[0])


    def gen_code(self,img): # -> code

        result_ret_list = []
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret , code_thres = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        ret ,_   = cv2.findContours(code_thres, cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)
        for i,r in enumerate( ret ) :
            if len(r) < 50 and len(r) > 20 :
                result_ret = i
                result_ret_list.append(result_ret)
        x = 0 
        y = 0 
        w = 0
        h = 0
        for result_ret in result_ret_list :
            nx,ny,nw,nh = cv2.boundingRect(ret[result_ret])
            x = (nx if x == 0 else min(x,nx))
            y = (ny if y == 0 else min(y,ny))
            w = (nw if w == 0 else max(w,nw))
            h = (nh if h == 0 else max(h,nh))
        split_img = code_thres[y:y+h,x:x+w]
        return self.get_code(split_img)

    def get_result(self,img_type):

        img = cv2.imdecode(np.frombuffer(img_type, np.uint8),1)
        _ , thres = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        one_code = np.zeros(thres.shape,dtype=np.uint8)
        for i in range(0,13): 
            for j in range(0,22):
                one_code[j][i] = thres[j][i]
        first_code = self.gen_code(one_code)
        two_sign = np.zeros(thres.shape,dtype=np.uint8)
        for i in range(13,20): 
            for j in range(0,22):
                two_sign[j][i] = thres[j][i]
        two_sign = self.gen_code(two_sign)
        three_code = np.zeros(thres.shape,dtype=np.uint8)
        for i in range(20,30): 
            for j in range(0,22):
                three_code[j][i] = thres[j][i]
        three_code = self.gen_code(three_code)
        if two_sign == "+" :
            result = int(first_code) + int(three_code)
        else :
            result = int(first_code) * int(three_code)

        return result
    