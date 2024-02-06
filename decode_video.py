# import av
# import av.datasets


# content = av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
# with av.open(content) as container:
#     # Signal that we only want to look at keyframes.
#     stream = container.streams.video[0]
#     stream.codec_context.skip_frame = "NONKEY"

#     for frame in container.decode(stream):
#         print(frame)

#         # We use `frame.pts` as `frame.index` won't make must sense with the `skip_frame`.
#         frame.to_image().save(
#             "night-sky.{:04d}.jpg".format(frame.pts),
#             quality=80,
#         )

import av
import librosa
import numpy as np
import av.datasets
input_ = av.open("sample3.mp4","r")
#codec = av.CodecContext.create("h264", "r")
output = av.open("remuxedaudio0_0.mkv", "w")
import time
import whisper

in_stream = input_.streams.get(audio=0)
print(in_stream[0].type)

#time.sleep(3)
out_stream = output.add_stream(template=in_stream[0])
#out_stream = output.add_stream(template=in_stream[1])
print(out_stream)
#time.sleep(3)
modelw = whisper.load_model("small")

i=0
total = 0
mono = []
for packet in input_.decode(in_stream):
    #print(dir(packet))
    #print(packet.time)
    # We need to skip the "flushing" packets that `demux` generates.
    #print(packet)
    #time.sleep(3)
    if packet.dts is None:
        continue
    tim = librosa.frames_to_time(packet.to_ndarray()[0]).astype(np.float32)
    #tim = packet.to_ndarray()[0]
    mono += list(tim)
    #print(len(mono))
    if len(mono)>48000:
        result = modelw.transcribe(np.array(mono),language='en')
        for it in result['segments']:
            print(f'Start:\t{it["start"]+total}\tEnd:\t{it["end"]+total}Text:\t{it["text"]}')
        mono = []
        #print(i)
        #time.sleep(3)
        #print(f'Start:\t{int(round(fps*it["start"])+index_segment*frames_interval)}\tEnd:\t{int(round(fps*it["end"])+index_segment*frames_interval)}Text:\t{it["text"]}')
    total+=packet.time


    # We need to assign the packet to the new stream.
    #packet.stream = out_stream

    #output.mux(packet)
    i+=1
print(i)

input_.close()
output.close()
# print(f'time_base:\t{in_stream.time_base}')
# print(f'start:\t\t{in_stream.start_time}')
# print(f'duration:\t{in_stream.duration}')
# print(f'frames:\t\t{in_stream.frames}')
# print(f'type:\t\t{in_stream.type}')
# print(f'codex:\t\t{in_stream.codec_context}')
# print(f'id:\t\t{in_stream.id}')
# print(f'index:\t\t{in_stream.index}')
# print(f'options:\t{in_stream.options}')
# print(f'avg rate:\t{in_stream.average_rate}')
# print(f'base rate:\t{in_stream.base_rate}')
# print(f'guess rate:\t{in_stream.guessed_rate}')
# print(f'profile:\t{in_stream.profile}')
# print(f'lang:\t{in_stream.language}')
# container = streams.StreamContainer()
# container.get(video=0)
# video = container.streams.video[0]
# audio = container.streams.get(audio=(0, 1))
# print(type(video))
