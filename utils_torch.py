import re 
import os
import cv2
import numpy as np
import time
import torch
import scipy
import copy
import librosa
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F


from sys import maxsize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

class Image_Caption():
    def __init__(self,
                images_dir,
                model,
                encoder,
                decoder,
                w2v,
                tier,
                word_index_Mapping,
                index_word_Mapping,
                word_map,
                max_caption_length,
                vocab_size,
                beam_size  ):
        
        self.images_dir = images_dir
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.w2v = w2v
        self.tier = tier
        self.word_index_Mapping = word_index_Mapping
        self.index_word_Mapping = index_word_Mapping
        self.word_map = word_map
        self.max_caption_length = max_caption_length
        self.vocab_size = vocab_size
        self.beam_size = beam_size


    def caption_video(self,filename,is_url:bool=False,url:str='https://youtu.be/oTN7xO6emU0'):
        if is_url:
            os.system(f'yt-dlp --verbose  --recode-video mp4 {url} -o {filename}')
        font = 0
        org = (50, 50) 
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
        

        cap = cv2.VideoCapture(filename)
 
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        # Read until video is completed
        while(cap.isOpened()):
            
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:

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

        diff = np.abs(p_hist - c_hist)

        return np.sum(diff)

    

    def get_caption_per_photo(self, img_bgr):
        """
        Reads an image and captions it with beam search.

        :param self.encoder: encoder model
        :param decoder: decoder model
        :param image_path: path to image
        :param word_map: word map
        :param beam_size: number of sequences to consider at each decode-step
        :return: caption, weights for visualization
        """

        k = self.beam_size
        vocab_size = len(self.word_map)

        # Read image and process
        #img_bgr = cv2.imread(image_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = cv2.resize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        img = img / 255.
        img = torch.FloatTensor(img).to(device)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])
        image = transform(img)  # (3, 256, 256)

        # Encode
        image = image.unsqueeze(0)  # (1, 3, 256, 256)
        encoder_out = self.encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = self.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

            gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, self.decoder_dim)

            scores = self.decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != self.word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

        return seq, alphas


    
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

