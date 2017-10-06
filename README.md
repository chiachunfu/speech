# TensorFlow on mobile with speech-to-text DL models.

For this three-week project at Insight, I worked on building a speech-to-text system that runs inference on Android devices. The project
can be divided into two parts: 

  1. Speech-to-text model
  2. Tensorflow inference on Android

For the first part, I took two major approaches: one is to build a small model from scratch and the other is to compress a 
pretrained model. Considering the model performace, I ended up deploying a [pretrained WaveNet model](https://github.com/buriburisuri/speech-to-text-wavenet) on Android.
But in this repository, I'll step-by-step walk you through both processes: deploy a pretrained WaveNet model on Android and build a small speech-to-text model with LSTM and CTC loss.

## Deploy a pretrained WaveNet model on Android

If you'd like to try out the app, everything you need are included in the 'speechandroid' folder. You can clone or download 
this repository and use Android Studio to open and run the project. Make sure that you have SDK and NDK installed (it can be 
done through Android Studio). If you'd like to build the app from start to the end, first let's:

### GET THE MODEL
Unlike image problems, it's not easy to find a pretrained DL model for speech-to-text that gives out checkpoints. Luckily, I found this
WaveNet speech-to-text implementation [here](https://github.com/buriburisuri/speech-to-text-wavenet). To export the model for compression, 
I ran the docker image, loaded the checkpoint and wrote it into a protocol buffers file by running
```python
python export_wave_pb.py
```
in the docker container. Next, we need to

### BAZEL BUILD TENSORFLOW FROM SOURCE AND QUANTIZE THE MODEL
In order to [quantize the model with Tensorflow](https://www.tensorflow.org/performance/quantization), you need to have bazel installed and clone [Tensorflow repository](https://github.com/tensorflow/tensorflow). I recommend
 creating a new virtual environment and bazel [build tensorflow](https://www.tensorflow.org/install/install_sources) there. Once it's done, you can run
```shell
bazel build tensorflow/tools/graph_transforms:transform_graph
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=/your/.pb/file \
  --outputs="output_node_name" \
  --out_graph=/the/quantized/.pb/file \
  --transforms='quantize_weights'
```
You can check out the official quantization tutorial on Tensorflow website for other options in 'transforms'. After quantization, the
model was sized down by 75% from 15.5Mb to 4.0Mb because of the eight-bit conversion. 

### CONVERT RAW AUDIO INTO MFCC


### TENSORFLOW OPS REGISTRATION 


### ANDRIOD APP


## Build a small speech-to-text model with LSTM and CTC loss

### Requirements


### Data processing


### Training


### Recognition


### Results



