import re 
import os
import cv2
import numpy as np
import time
import scipy
import copy
import librosa
import pydub
import matplotlib.pyplot as plt
import shutil

from sys import maxsize
from openai import OpenAI
import whisper
from pydub import AudioSegment
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.applications.inception_v3 import preprocess_input

KP_THRESHOLD = 30

def preprocess(raw_text):

    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    words = letters_only_text.lower().split()

    return words

def get_tier(filename):
    with  open(filename, "r") as my_file:
        data = my_file.read() 
        data_list = data.split('\n')
    return data_list

def chunks_video(filename):
    vid = AudioSegment.from_file(filename,'mp4')
    chunk_length = 2000 # in ms
    chunk_max = (len(vid)//chunk_length) +1
    print(f'Length video: {len(vid)}')
    print(f'Chunks video: {chunk_max}')

    folder_path = 'temp_' + filename.split('/')[-1]
    os.mkdir(folder_path)
    for i in range(chunk_max):
        chunk = vid[i*chunk_length:(i+1)*chunk_length]
        chunk.export(f'{folder_path}/{i}.mp3',format='mp3')
    return folder_path, int(chunk_length/1000)  # in sec

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

    def caption_video(self,filename,is_url:bool=False,url:str='https://youtu.be/oTN7xO6emU0',subtitles:bool = False):
        if is_url:
            os.system(f'yt-dlp --verbose  --recode-video mp4 {url} -o {filename}')
        client = OpenAI()

        font = 0
        org = (50, 50) 
        org2 = (50,550)
        fontScale = 1
        color = (255, 0, 0) 
        color2 = (0, 0, 255) 
        thickness = 2
        frame_cnt=-1
        mw = 64
        mh = 36
        mstack = 10
        vis_frames = np.zeros((mh,mw*mstack,3),dtype=np.uint8)
        matches = 0
        caption = 'captplaceholder'
        folder_chunks = None
        chunk_len = None
        chunk_files = []
        subtitle_text = ''
        model = whisper.load_model("base")

        cap = cv2.VideoCapture(filename)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

        print(f"{fps} frames per second")

 
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        if subtitles:
            folder_chunks, chunk_len = chunks_video(filename)
            frames_interval = fps*chunk_len
            print(f'Frames interval:\t{frames_interval}')
            chunk_files = [f'{folder_chunks}/{f}' for f in os.listdir(folder_chunks)]

        # Read until video is completed
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                #img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                # ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                # #th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
                # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                # #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                # blur = cv2.GaussianBlur(img,(7,7),0)
                # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                # titles = ['Original Image', 'Global Thresholding (v = 127)',
                #             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
                # images = [frame, th1, th2, th3]
                # for i in range(4):
                #     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
                #     plt.title(titles[i])
                #     plt.xticks([]),plt.yticks([])
                # plt.show()
                # thresholds = threshold_multiotsu(img,classes=5)
                # # Generate regions
                # regions = np.digitize(img, bins=thresholds)
                # plt.imshow(regions, cmap='jet')
                # plt.show()
                if frame_cnt%frames_interval==0 and subtitles:
                    #os.system(f'whisper --model base "{chunk_files[frame_cnt//frames_interval]}" --language=en')
                    # audio_file= open(chunk_files[frame_cnt//frames_interval], "rb")
                    # transcript = client.audio.transcriptions.create(
                    # model="whisper-1", 
                    # file=audio_file
                    # )
                    result = model.transcribe(f'{chunk_files[frame_cnt//frames_interval]}')
                    print(result["text"])

                    subtitle_text = result["text"]
                    print(subtitle_text)

                frame_cnt +=1

                # Display the resulting frame
                #
                mini_frame = cv2.cvtColor(cv2.resize(frame,(mw,mh)), cv2.COLOR_BGR2RGB)

                if frame_cnt==0:
                    pass
                else:
                    matches = self.get_matching_points(cv2.resize(prev,(256,144)),cv2.resize(frame,(256,144)))
                if matches == False or frame_cnt==0:
                    print(f'\n{frame_cnt}.')
                    caption = self.get_caption_per_photo(frame)
                    scores,(bsc,best_score) = self.scores_per_class(caption)
                    mini_frame[:,:5,0] = 255
                    mini_frame[:,:5,1] = 0
                    mini_frame[:,:5,2] = 255
                
                    print(caption)
                    print(f'\nBest score-> {bsc}:\t{best_score}')
                if frame_cnt%mstack==0 and frame_cnt!=0:
                    plt.figure(figsize = (17,2))
                    plt.axis('off')
                    plt.imshow(vis_frames)
                    plt.show()
                    vis_frames*=0
                indx = (frame_cnt%mstack)
                vis_frames[:,indx*mw:(indx+1)*mw,:] += mini_frame

                frame = cv2.putText(frame, f'{caption}', org, font,  
                    fontScale, color, thickness, cv2.LINE_AA)
                if subtitles:
                    frame = cv2.putText(frame, f'{subtitle_text}', org2, font,  
                        fontScale, color2, thickness, cv2.LINE_AA)
                cv2.imshow('Frame',frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

                prev = copy.deepcopy(frame)
                            
        # Break the loop

            else: 
                break
        
        # When everything done, release the video capture object
        cap.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()

        if folder_chunks is not None:
            shutil.rmtree(folder_chunks)

    def cosine_distance_between_two_words(self,word1, word2):
        return (1- scipy.spatial.distance.cosine(self.w2v[word1], self.w2v[word2]))

    def cosine_distance_wordembedding_method(self,s1, s2):
        vector_1 = np.mean([self.w2v[word] for word in preprocess(s1)],axis=0)
        vector_2 = np.mean([self.w2v[word] for word in preprocess(s2)],axis=0)
        cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
        #print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
        return round((1-cosine)*100,3)

    def scores_per_class(self,caption):
        scores = {}
        best_score = 0
        best_score_class = ''
        for c in self.tier:
            scores[c] = self.cosine_distance_wordembedding_method(caption,c)
            if scores[c]> best_score:
                best_score = scores[c]
                best_score_class = c
        return scores , (best_score_class,best_score)
    
    
    def get_matching_points(self,prev_frame,current_frame):
        ismatch = True
        t0 = time.time()
        prev_frame = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
        current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
        orb = cv2.SIFT_create()
        THRESHOLD_FEATURES = 40
        
        try:
            kp1, des1 = orb.detectAndCompute(prev_frame,None)
            kp2, des2 = orb.detectAndCompute(current_frame,None)

            if len(kp1)<THRESHOLD_FEATURES or len(kp2)<THRESHOLD_FEATURES:
                diff = self.compare_frames(prev_frame,current_frame)
                if diff < 50000/(256*144):
                    return True
                else:
                    print(f'Diff: {diff}')
                    return False

            bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

            matches = bf.knnMatch(des1,des2,k=2)

            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])        

            #print(f'Time passed kp: {time.time()-t0}')
            if len(good) < KP_THRESHOLD:
                ismatch=False

            return ismatch
        except Exception as E:
            print(f'Exception: {E}')
            print(f'Error in kp: {len(kp1)} - {len(kp2)}')
            return 0
        
    def compare_frames(self,prev_frame,current_frame):
        t0 =time.time()
        p_hist,p_bins= np.histogram(prev_frame, bins=256, range=(0,255),density=True)
        c_hist,p_bins= np.histogram(current_frame, bins=256, range=(0,255),density=True)
        # plt.title('histos')
        # plt.plot(np.arange(256),p_hist,'r',np.arange(256),c_hist,'b')
        # plt.show()
        diff = np.abs(p_hist - c_hist)
        # plt.title(np.sum(diff))
        # plt.hist(diff,256,[0,256])
        # plt.show()
        return np.sum(diff)

    

    
def get_key(audio_file_path):

    y, sr = librosa.load(audio_file_path)

    # Compute the Chroma Short-Time Fourier Transform (chroma_stft)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

    # Calculate the mean chroma feature across time
    mean_chroma = np.mean(chromagram, axis=1)

    # Define the mapping of chroma features to keys
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Find the key by selecting the maximum chroma feature
    estimated_key_index = np.argmax(mean_chroma)
    estimated_key = chroma_to_key[estimated_key_index]

    # Print the detected key
    print("Detected Key:", estimated_key)

