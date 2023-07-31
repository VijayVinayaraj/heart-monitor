
#include ".h"


#include "TensorFlowLite.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"


extern "C" {
#include "utils.h"
};


#define DEBUG 1


constexpr int BUZZER_PIN = A1;


constexpr int NUM_AXES = 3;         
constexpr int MAX_MEASUREMENTS = 128; 
constexpr float MAD_SCALE = 1.4826;   
constexpr float THRESHOLD = 2e-05;   
constexpr int WAIT_TIME = 1000;       
constexpr int SAMPLE_RATE = 200;    


void getHeartbeat();
void detectAnomaly(int mse);

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

 
  constexpr int kTensorArenaSize = 1 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} 
 

LowPassFilter low_pass_filter_red(kLowPassCutoff, kSamplingFrequency);
LowPassFilter low_pass_filter_ir(kLowPassCutoff, kSamplingFrequency);
HighPassFilter high_pass_filter(kHighPassCutoff, kSamplingFrequency);
Differentiator differentiator(kSamplingFrequency);
MovingAverageFilter<kAveragingSamples> averager_bpm;
MovingAverageFilter<kAveragingSamples> averager_r;
MovingAverageFilter<kAveragingSamples> averager_spo2;


MinMaxAvgStatistic stat_red;
MinMaxAvgStatistic stat_ir;

float kSpO2_A = 1.5958422;
float kSpO2_B = -34.6596622;
float kSpO2_C = 112.6898759;


long last_heartbeat = 0;


long finger_timestamp = 0;
bool finger_detected = false;

float last_diff = NAN;
bool crossed = false;
long crossed_time = 0;
 
void setup() {
 Serial.begin(115200);
pinMode(buzzer,OUTPUT);

  digitalWrite(buzzer,LOW);


  if(sensor.begin() && sensor.setSamplingRate(kSamplingRate)) { 
    Serial.println("Sensor initialized");
  }
  else {
    Serial.println("Sensor not found");  
    while(1);
  }

  }


  pinMode(BUZZER_PIN, OUTPUT);


  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(fan_low_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }


  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_FULLY_CONNECTED,
    tflite::ops::micro::Register_FULLY_CONNECTED(),
    1, 3);

  static tflite::MicroInterpreter static_interpreter(
    model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }


  model_input = interpreter->input(0);
  model_output = interpreter->output(0);


  
  
}

void loop() {
    getHeartbeat();
  float sample[MAX_MEASUREMENTS][NUM_AXES];
  float measurements[MAX_MEASUREMENTS];
  float mad[NUM_AXES];
  float y_val[NUM_AXES];
  float mse;
  TfLiteStatus invoke_status;
  

  static unsigned long timestamp = millis();
  static unsigned long prev_timestamp = timestamp;



  

  for (int axis = 0; axis < NUM_AXES; axis++) {
    for (int i = 0; i < MAX_MEASUREMENTS; i++) {
      measurements[i] = sample[i][axis];
    }
    mad[axis] = MAD_SCALE * calc_mad(measurements, MAX_MEASUREMENTS);
  }





  for (int axis = 0; axis < NUM_AXES; axis++) {
    model_input->data.f[axis] = mad[axis];
  }


  invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input");
  }


  for (int axis = 0; axis < NUM_AXES; axis++) {
    y_val[axis] = model_output->data.f[axis];
  }

  mse = calc_mse(mad, y_val, NUM_AXES);

 detectAnomaly(mse);

  delay(WAIT_TIME);

}

void detectAnomaly(int mse){
if (mse > THRESHOLD) {
    digitalWrite(BUZZER_PIN, HIGH);

    Serial.println("Anomaly");

  } else {
    digitalWrite(BUZZER_PIN, LOW);
  Serial.println("Normal");
  }



}


void getHeartbeat(){
    auto sample = sensor.readSample(1000);
  float current_value_red = sample.red;
  float current_value_ir = sample.ir;
  

  if(sample.red > kFingerThreshold) {
    if(millis() - finger_timestamp > kFingerCooldownMs) {
      finger_detected = true;
    }
  }
  else {
 
    differentiator.reset();
    averager_bpm.reset();
    averager_r.reset();
    averager_spo2.reset();
    low_pass_filter_red.reset();
    low_pass_filter_ir.reset();
    high_pass_filter.reset();
    stat_red.reset();
    stat_ir.reset();
    
    finger_detected = false;
    finger_timestamp = millis();
  }

  if(finger_detected) {
    current_value_red = low_pass_filter_red.process(current_value_red);
    current_value_ir = low_pass_filter_ir.process(current_value_ir);


    stat_red.process(current_value_red);
    stat_ir.process(current_value_ir);


    float current_value = high_pass_filter.process(current_value_red);
    float current_diff = differentiator.process(current_value);


    if(!isnan(current_diff) && !isnan(last_diff)) {
      

      if(last_diff > 0 && current_diff < 0) {
        crossed = true;
        crossed_time = millis();
      }
      
      if(current_diff > 0) {
        crossed = false;
      }
  

      if(crossed && current_diff < kEdgeThreshold) {
        if(last_heartbeat != 0 && crossed_time - last_heartbeat > 300) {
          int bpm = 60000/(crossed_time - last_heartbeat);
          float rred = (stat_red.maximum()-stat_red.minimum())/stat_red.average();
          float rir = (stat_ir.maximum()-stat_ir.minimum())/stat_ir.average();
          float r = rred/rir;
          float spo2 = kSpO2_A * r * r + kSpO2_B * r + kSpO2_C;
          
          if(bpm > 50 && bpm < 250) {

            if(kEnableAveraging) {
              int average_bpm = averager_bpm.process(bpm);
              int average_r = averager_r.process(r);
              int average_spo2 = averager_spo2.process(spo2);
  

              if(averager_bpm.count() >= kSampleThreshold) {
    
                  Serial.println(average_bpm);

              }
            }
            else {
     
            detectAnomaly(bpm);
            }
          }

          stat_red.reset();
          stat_ir.reset();
        }
  
        crossed = false;
        last_heartbeat = crossed_time;
      }
    }

    last_diff = current_diff;
  }
}