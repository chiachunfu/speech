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
In order to [quantize the model with Tensorflow](https://www.tensorflow.org/performance/quantization), you need to have bazel installed and clone [Tensorflow repository](https://github.com/tensorflow/tensorflow). I recommend creating a new virtual environment and bazel [build tensorflow](https://www.tensorflow.org/install/install_sources) there. Once it's done, you can run:
```shell
bazel build tensorflow/tools/graph_transforms:transform_graph
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=/your/.pb/file \
  --outputs="output_node_name" \
  --out_graph=/the/quantized/.pb/file \
  --transforms='quantize_weights'
```
You can check out the official quantization tutorial on Tensorflow website for other options in 'transforms'. After quantization, the model was sized down by 75% from 15.5Mb to 4.0Mb because of the eight-bit conversion. Due to the time limit, I haven't calculated the letter error rate with a test set to quantify the accuracy drop before and after quantization. For detailed discussion on neural network quantization, [here](https://petewarden.com/2017/06/22/what-ive-learned-about-neural-network-quantization/) is a great post by Pete Warden. So now we have a compressed pretrained model, let's see what else we need to deploy the model on Android:

### TENSORFLOW OPS REGISTRATION 
Here we need to bazel build tensorflow to create a .so file that can be called by JNI and includes all the operation libraries we need for the pretrained WaveNet model inference. First, let's edit the WORKSPACE file in the cloned TensorFlow repository by uncommenting and updating the paths to SDK and NDK. Second, we need to find out what ops were used in the pretrained model and generate a .so file with that piece of information. There are two ways to do this (only the second one worked for me):
  1. Use [selective registration](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/print_selective_registration_header.py)
  
     First run:
     ``` 
      bazel build tensorflow/python/tools:print_selective_registration_header && \
      bazel-bin/tensorflow/python/tools/print_selective_registration_header \
      --graphs=path/to/graph.pb > ops_to_register.h
      ```
     All the ops in the .pb file will be listed out in ops_to_register.h. 
     Next, move op_to_register.h to /tensorflow/tensorflow/core/framework/ and run:
     
     ```
      bazel build -c opt --copt="-DSELECTIVE_REGISTRATION" \
      --copt="-DSUPPORT_SELECTIVE_REGISTRATION" \
      //tensorflow/contrib/android:libtensorflow_inference.so \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      --crosstool_top=//external:android/crosstool --cpu=armeabi-v7a
      ```
     Unfortunately, while I didn't get any error message, the .so file still didn't include all the ops listed in the header file. 
    
  2. Modify BUILD in /tensorflow/tensorflow/core/kernels/
  
     If you didn't try out the first option and get the list of ops in the model, you can get that by tf.train.write_graph your graph and [do](https://github.com/tensorflow/tensorflow/issues/3549):
     ```
     grep "op: " PATH/TO/mygraph.txt | sort | uniq | sed -E 's/^.+"(.+)".?$/\1/g'
     ```
     in your terminal. Next, edit the BUILD file by adding the missing ops into the 'android_extended_ops_group1' or 'android_extended_ops_group2' in the Android libraries section. You can also make the .so file smaller by removing unneeded ops. Now, run:
     ```
     bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so \
     --crosstool_top=//external:android/crosstool \
     --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
     --cpu=armeabi-v7a
     ```
     And you'll find the libtensorflow_inference.so file in:
     ```
     bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so
     ```
  
**NOTE** I ran into an error with the sparse_to_dense op when running on Android. If you'd like to repeat this work, add 'REGISTER_KERNELS_ALL(int64);' to sparse_to_dense_op.cc, line 153.

In addition to .so file, we also need a JAR file. You can simply add this in [the build.gradle file](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android):

```
      allprojects {
        repositories {
            jcenter()
        }
      }

      dependencies {
            compile 'org.tensorflow:tensorflow-android:+'
      }
```
or you can run:
   ```
      bazel build //tensorflow/contrib/android:android_tensorflow_inference_java
   ```
and you'll find the file at:
   ```
      bazel-bin/tensorflow/contrib/android/libandroid_tensorflow_inference_java.jar
   ```
Now, move both files to your android project. 

### CONVERT RAW AUDIO INTO MEL-FREQUENCY CEPSTRAL COEFFICIENTS (MFCC)
As the pretrained WaveNet is traied with [MFCC](http://recognize-speech.com/feature-extraction/mfcc) inputs, we need to add this feature extraction method into our pipeline. The source-build TensorFlow has an audio op that can perform this feature extraction. My initial thought was to wrap this operation with the pretrained wavenet and I did it by using a trick I found [here](https://stackoverflow.com/questions/43332342/is-it-possible-to-replace-placeholder-with-a-constant-in-an-existing-graph/43342922#43342922).  It turned out that there are some variations in how one can convert raw audio into MFCC. As shown below, the MFCC from Tensorflow audio op is different from the one given by librosa, a python library used by the pretrained WaveNet authors for converting training data into MFCC:

<p align="center">
  <img src="https://github.com/chiachunfu/speech/blob/master/MFCC.png">
</p>

Now wrapping the TensorFlow operation into the model is out of the picture. To make this work, I rewrote the librosa MFCC feature with Java so I could add the function between the raw audio input and the model in the Android app. The MFCC.java file can be found in /speechandroid/src/org/tensorflow/demo/mfcc/. 

### ANDRIOD APP
I modified the TF speech example in [Tensorflow Android Demo repository](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android). The build.gradle file in the demo actually helps you build the .so and jar file. So if you'd like to start with demo example with your own model, you can simply get the list of your ops, modify the BUILD file and let the build.gradle file take care of the rest.  


## Build a small speech-to-text model with LSTM and CTC loss

### REQUIREMENTS
* tensorflow
* python
* numpy
* python-speech-features

### REFERENCE
The basic script and the file_logger.py and constants.py are borrowed from [here](https://github.com/philipperemy/tensorflow-ctc-speech-recognition) and [here](https://github.com/igormq/ctc_tensorflow_example).

### DATA PROCESSING
I used the [VCTK corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html). After downloading the data set, please delete p315 as the txt files are missing. See 'Data_Process.ipynb' for splitting data into train/dev/test sets, doing training data normilzation and pickling data into .npz files. I also got rid of files (after pickled) that are less than 10kb because some of the wav files less than that size are corrupted. In order to speed up the batch training, I only trained and validated on .npz files that are between 10kb and 70kb.

### TRAINING
The number of hidden units, batch size and the path to the data folders can be updated in the conf.json file. The number of MFCC features should be adjusted depending on how you preprocess the wav files. I implemented the batch training and added gradient clipping to make the training run properly. You can start the training by:

```python
    python lstm_ctc.py
```
You can use Tensorboard to load the summary and check your performance, then export your selected model with:
```python
    python export_lstm_pb.py
```
Next do inference with .npz file (or use librosa and python-speech-features to read and transform wav_files):
```python
    python lstm_pb_recognize.py
```

### WHAT'S NEXT?






