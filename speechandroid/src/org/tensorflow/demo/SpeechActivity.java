/*
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* The backbone of this app is based on the tutorial below, but I used a pretrained
   speech-to-text model instead.

   https://www.tensorflow.org/tutorials/audio_training

   The model files can be find in the 'assets' folder.
*/

package org.tensorflow.demo;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.util.Arrays;
import java.util.concurrent.locks.ReentrantLock;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.mfcc.MFCC;


/**
 * An activity that listens for audio and then uses a Pretrained model to detect speech content,
 * and turn it into text.
 */
public class SpeechActivity extends Activity {

  // Constants that control the behavior of the recognition code and model
  // settings.
  private static final int SAMPLE_RATE = 16000;
  private static final int SAMPLE_DURATION_MS = 5000;
  private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
  private static final String MODEL_FILENAME = "file:///android_asset/q_wavenet_mobile.pb";
  private static final String INPUT_DATA_NAME = "Placeholder:0";
  private static final String OUTPUT_SCORES_NAME = "output";

    private static final char[] map = new char[]{'0', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

  // UI elements.
  private static final int REQUEST_RECORD_AUDIO = 13;
  private Button startButton;
    private TextView outputText;
  private static final String LOG_TAG = SpeechActivity.class.getSimpleName();

  // Working variables.
  short[] recordingBuffer = new short[RECORDING_LENGTH];
  int recordingOffset = 0;
  boolean shouldContinue = true;
  private Thread recordingThread;
  boolean shouldContinueRecognition = true;
  private Thread recognitionThread;
  private final ReentrantLock recordingBufferLock = new ReentrantLock();
  private TensorFlowInferenceInterface inferenceInterface;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    // Set up the UI.
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_speech);
    startButton = (Button) findViewById(R.id.start);
    startButton.setOnClickListener(
        new View.OnClickListener() {
          @Override
          public void onClick(View view) {

              startRecording();
          }
        });
    outputText = (TextView) findViewById(R.id.output_text);


    // Load the Pretrained WaveNet model.
    inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILENAME);

    requestMicrophonePermission();
  }

  private void requestMicrophonePermission() {
    requestPermissions(
        new String[] {android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_RECORD_AUDIO
        && grantResults.length > 0
        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
    }
  }

  public synchronized void startRecording() {
    if (recordingThread != null) {
      return;
    }
    shouldContinue = true;
    recordingThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                record();
              }
            });
    recordingThread.start();
  }

  private void record() {
    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

    // Estimate the buffer size we'll need for this device.
    int bufferSize =
        AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
    if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
      bufferSize = SAMPLE_RATE * 2;
    }
    short[] audioBuffer = new short[bufferSize / 2];

    AudioRecord record =
        new AudioRecord(
            MediaRecorder.AudioSource.DEFAULT,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize);

    if (record.getState() != AudioRecord.STATE_INITIALIZED) {
      Log.e(LOG_TAG, "Audio Record can't initialize!");
      return;
    }

    record.startRecording();

    Log.v(LOG_TAG, "Start recording");


    while (shouldContinue) {
      int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
        Log.v(LOG_TAG, "read: " + numberRead);
      int maxLength = recordingBuffer.length;
      recordingBufferLock.lock();
      try {
          if (recordingOffset + numberRead < maxLength) {
              System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, numberRead);
          } else {
              shouldContinue = false;
          }
        recordingOffset += numberRead;
      } finally {
        recordingBufferLock.unlock();
      }
    }
    record.stop();
    record.release();
      startRecognition();
  }

  public synchronized void startRecognition() {
    if (recognitionThread != null) {
      return;
    }
    shouldContinueRecognition = true;
    recognitionThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                recognize();
              }
            });
    recognitionThread.start();
  }

  private void recognize() {
    Log.v(LOG_TAG, "Start recognition");

    short[] inputBuffer = new short[RECORDING_LENGTH];
    double[] doubleInputBuffer = new double[RECORDING_LENGTH];
    long[] outputScores = new long[157];
    String[] outputScoresNames = new String[]{OUTPUT_SCORES_NAME};


      recordingBufferLock.lock();
      try {
        int maxLength = recordingBuffer.length;
          System.arraycopy(recordingBuffer, 0, inputBuffer, 0, maxLength);
      } finally {
        recordingBufferLock.unlock();
      }

      // We need to feed in float values between -1.0 and 1.0, so divide the
      // signed 16-bit inputs.
      for (int i = 0; i < RECORDING_LENGTH; ++i) {
        doubleInputBuffer[i] = inputBuffer[i] / 32767.0;
      }

      //MFCC java library.
      MFCC mfccConvert = new MFCC();
      float[] mfccInput = mfccConvert.process(doubleInputBuffer);
      Log.v(LOG_TAG, "MFCC Input======> " + Arrays.toString(mfccInput));

      // Run the model.
      inferenceInterface.feed(INPUT_DATA_NAME, mfccInput, 1, 157, 20);
      inferenceInterface.run(outputScoresNames);
      inferenceInterface.fetch(OUTPUT_SCORES_NAME, outputScores);
      Log.v(LOG_TAG, "OUTPUT======> " + Arrays.toString(outputScores));


      //Output the result.
      String result = "";
      for (int i = 0;i<outputScores.length;i++) {
          if (outputScores[i] == 0)
              break;
          result += map[(int) outputScores[i]];
      }
      final String r = result;
      this.runOnUiThread(new Runnable() {
          @Override
          public void run() {
              outputText.setText(r);
          }
      });

      Log.v(LOG_TAG, "End recognition: " +result);
    }

}
