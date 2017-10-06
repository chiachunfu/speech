# TensorFlow on mobile with speech-to-text DL models.

For this three-week project at Insight, I worked on building a speech-to-text system that runs inference on Android devices. The project can be divided into two parts: 

  1. Speech-to-text model
  2. Tensorflow inference on Android

For the first part, I took two major approaches: one is to build a small model from scratch and the other is to compress a 
pretrained model. Considering the model performace, I ended up deploying a [pretrained WaveNet model](https://github.com/buriburisuri/speech-to-text-wavenet) on Android.
In this repository, I'll step-by-step walk you through both processes: deploy a pretrained WaveNet model on Android and build a small speech-to-text model with LSTM and CTC loss.

## Deploy a pretrained WaveNet model on Android

### ENVIRONMENT INFO
  * MacOS Sierra 10.12.6
  * Android 7.1.1
  * Android NDK 15.2
  * Android gradle plugin 2.3.0
  * TensorFlow 1.3.0
  * bazel 0.5.4-homebrew

### TRY THE APP
If you'd like to try out the app, everything you need are included in the 'speechandroid' folder. You can clone or download 
this repository and use Android Studio to open and run the project. Make sure that you have SDK and NDK installed (it can be 
done through Android Studio). If you'd like to build the app from start to the end, first let's:

### GET THE MODEL
Unlike image problems, it's not easy to find a pretrained DL model for speech-to-text that gives out checkpoints. Luckily, I found this WaveNet speech-to-text implementation [here](https://github.com/buriburisuri/speech-to-text-wavenet). To export the model for compression, I ran the docker image, loaded the checkpoint and wrote it into a protocol buffers file by running
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
You can check out the official quantization tutorial on Tensorflow website for other options in 'transforms'. After quantization, the model was sized down by 75% from 15.5Mb to 4.0Mb because of the eight-bit conversion. So now we have a compressed pretrained model, and let's see what else we need to deploy the model on Android:

### TENSORFLOW OPS REGISTRATION 


### CONVERT RAW AUDIO INTO MEL-FREQUENCY CEPSTRAL COEFFICIENTS (MFCC)
As the pretrained WaveNet is traied with [MFCC](http://recognize-speech.com/feature-extraction/mfcc) inputs, we need to add this feature extraction method into our pipeline. The source-build TensorFlow has an audio op that can perform this feature extraction. My initial thought was to wrap this operation with the pretrained wavenet and I did it by using a trick I found [here](https://stackoverflow.com/questions/43332342/is-it-possible-to-replace-placeholder-with-a-constant-in-an-existing-graph/43342922#43342922).  It turned out that there are some variations in how one can convert raw audio into MFCC. As shown below, the MFCC from Tensorflow audio op is different from the one given by librosa, a python library used by the pretrained WaveNet authors for converting training data into MFCC:

![Image of MFCC](https://github.com/chiachunfu/speech/MFCC.png)

Now wrapping the TensorFlow operation into the model is out of the picture. To make this work, I rewrote the librosa MFCC feature with Java so I could add the function between the raw audio input and the model in the Android app. The MFCC.java file can be found in /speechandroid/src/org/tensorflow/demo/mfcc/. 

### ANDRIOD APP




## Build a small speech-to-text model with LSTM and CTC loss

### REQUIREMENTS


### DATA PROCESSING


### TRAINING


### RECOGNITION


### RESULTS






