import re 
import os
import cv2
import numpy as np
import time
import scipy
import matplotlib.pyplot as plt


from sys import maxsize
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.applications.inception_v3 import preprocess_input

def preprocess(raw_text):

    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    words = letters_only_text.lower().split()

    return words

def get_tier(filename):
    with  open(filename, "r") as my_file:
        data = my_file.read() 
        data_list = data.split('\n')
    return data_list

class Image_Caption():
    def __init__(self,
                images_dir,
                model,
                fe_model,
                w2v,
                tier,
                word_index_Mapping,
                index_word_Mapping,
                max_caption_length,
                vocab_size):
        
        self.images_dir = images_dir
        self.model = model
        self.fe_model = fe_model
        self.w2v = w2v
        self.tier = tier
        self.word_index_Mapping = word_index_Mapping
        self.index_word_Mapping = index_word_Mapping
        self.max_caption_length = max_caption_length
        self.vocab_size = vocab_size

                      
    def get_caption_per_photo(self,img):
        # imageFile = os.path.join(self.images_dir,filename)
        # img = cv2.imread(imageFile)
        # plt.imshow(img)
        # plt.show()
        
        fimg = self.feature_extractor(img)
        photo_feature = np.reshape(fimg,(1,2048))
        in_text = 'startSeq'
        for i in range(1,self.max_caption_length):
            seq = [self.word_index_Mapping[w] for w in in_text.split() if w in self.word_index_Mapping]
            in_seq = pad_sequences([seq],maxlen=self.max_caption_length)
            inputs = [photo_feature,in_seq]
            yhat = self.model.predict(x=inputs,verbose=0)
            yhat = np.argmax(yhat)
            word = self.index_word_Mapping[yhat]
            in_text += ' ' + word
            if word =='endSeq':
                break
        
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)

        return final
    

    def feature_extractor(self,frame):
        t0 = time.time()
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(299,299))
        img = np.expand_dims(img,axis=0)
        img = preprocess_input(img)
        vector = self.fe_model.predict(img,verbose=0)
        vector = np.reshape(vector,vector.shape[1])
        print(f'Time passed: {time.time()-t0}')
        return vector

    def caption_video(self,filename):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (50, 50) 
        fontScale = 1
        color = (255, 0, 0) 
        color2 = (0, 0, 255) 
        thickness = 2
        frame_cnt=0
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

        cap = cv2.VideoCapture(filename)
 
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        # Read until video is completed
        while(cap.isOpened()):
            
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                frame_cnt +=1
                if frame_cnt%5==0:
                    # Display the resulting frame
                    caption = self.get_caption_per_photo(frame)
                    frame = cv2.putText(frame, caption, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA)
                    fgmask = fgbg.apply(frame)
                    idx=(fgmask==0)
                    chx = np.zeros_like(frame)
                    chx[idx]=frame[idx]
                    no_zeros = chx[np.where(chx==0)]
                    chx = cv2.putText(chx, str(no_zeros.size), (150,100), font,  
                    fontScale, color2, thickness, cv2.LINE_AA)
                    cv2.imshow('Frame',chx)
                    scores,(bsc,best_score) = self.scores_per_classs(caption)
                    print(caption)
                    print(f'\nBest score-> {bsc}:\t{best_score}')
                    print(f'no zeros:\t{no_zeros}')
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(1000) & 0xFF == ord('q'):
                        break
                else:
                    cv2.imshow('Frame',frame)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
                            
        # Break the loop
            else: 
                break
        
        # When everything done, release the video capture object
        cap.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()

    def cosine_distance_between_two_words(self,word1, word2):
        return (1- scipy.spatial.distance.cosine(self.w2v[word1], self.w2v[word2]))

    def cosine_distance_wordembedding_method(self,s1, s2):
        vector_1 = np.mean([self.w2v[word] for word in preprocess(s1)],axis=0)
        vector_2 = np.mean([self.w2v[word] for word in preprocess(s2)],axis=0)
        cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
        #print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
        return round((1-cosine)*100,3)

    def scores_per_classs(self,caption):
        scores = {}
        best_score = 0
        best_score_class = ''
        for c in self.tier:
            scores[c] = self.cosine_distance_wordembedding_method(caption,c)
            if scores[c]> best_score:
                best_score = scores[c]
                best_score_class = c
        return scores , (best_score_class,best_score)