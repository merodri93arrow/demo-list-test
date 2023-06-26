/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/profiling/profiler.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <unistd.h>
#include <getopt.h>
#include <sys/time.h>
#include <stdio.h>
#include <fstream>

using namespace std;
using namespace cv;


#define LOG(x) std::cerr

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

struct Object{
    cv::Rect rec;
    int      class_id;
    float    prob;
};
struct Settings {
  int accel = 0;
};

Settings s;

float expit(float x) {
    return 1.f / (1.f + expf(-x));
}


//nms
float iou(Rect& rectA, Rect& rectB)
{
    int x1 = std::max(rectA.x, rectB.x);
    int y1 = std::max(rectA.y, rectB.y);
    int x2 = std::min(rectA.x + rectA.width, rectB.x + rectB.width);
    int y2 = std::min(rectA.y + rectA.height, rectB.y + rectB.height);
    int w = std::max(0, (x2 - x1 + 1));
    int h = std::max(0, (y2 - y1 + 1));
    float inter = w * h;
    float areaA = rectA.width * rectA.height;
    float areaB = rectB.width * rectB.height;
    float o = inter / (areaA + areaB - inter);
    return (o >= 0) ? o : 0;
}

void nms(vector<Object>& boxes,  const double nms_threshold)
{
    vector<int> scores;
    for(int i = 0; i < boxes.size();i++){
	scores.push_back(boxes[i].prob);
    } 
    vector<int> index;
    for(int i = 0; i < scores.size(); ++i){
        index.push_back(i);
    }
    sort(index.begin(), index.end(), [&](int a, int b){ return scores[a] > scores[b]; }); 
    vector<bool> del(scores.size(), false);
    for(size_t i = 0; i < index.size(); i++){
        if( !del[index[i]]){
            for(size_t j = i+1; j < index.size(); j++){
                if(iou(boxes[index[i]].rec, boxes[index[j]].rec) > nms_threshold){
                    del[index[j]] = true;
                }
            }
        }
    }
    vector<Object> new_obj;
    for(const auto i : index){
	Object obj;
	if(!del[i])
	{
	    obj.class_id = boxes[i].class_id;
	    obj.rec.x =  boxes[i].rec.x;
	    obj.rec.y =  boxes[i].rec.y;
	    obj.rec.width =  boxes[i].rec.width;
	    obj.rec.height =  boxes[i].rec.height;
	    obj.prob =  boxes[i].prob;		
	}
	new_obj.push_back(obj);
    }
    boxes = new_obj;
}

TfLiteDelegatePtrMap GetDelegates(Settings* s) {
  TfLiteDelegatePtrMap delegates;

  if (s->accel) {
    auto delegate = tflite::evaluation::CreateNNAPIDelegate();
    if (!delegate) {
      LOG(INFO) << "NNAPIDelegate acceleration is unsupported on this platform.";
    } else {
      delegates.emplace("NNAPI", std::move(delegate));
    }
  }
  return delegates;
}

void run_inference() {
 
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("/home/root/.cache/eiq/eIQVideoSwitchCore/model/ssd_mobilenet_v1_1_default_1.tflite");
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    
    // Resize input tensors, if desired.
    TfLiteTensor* output_locations = nullptr;
    TfLiteTensor* output_classes = nullptr;
    TfLiteTensor* num_detections = nullptr;
    
    int frames = 0;
    int frames_index = 0;

    auto cam = cv::VideoCapture("/home/root/.cache/eiq/eIQVideoSwitchCore/media/video_device.mp4");
    frames = cam.get(cv::CAP_PROP_FRAME_COUNT);
	
    std::vector<std::string> labels;
    
    auto file_name="/home/root/.cache/eiq/eIQVideoSwitchCore/model/labels.txt";
    std::ifstream input( file_name );
    for( std::string line; getline( input, line ); )
    {
        labels.push_back( line);
    }
    
    //interpreter->UseNNAPI(s.accel);
    auto delegates_ = GetDelegates( &s);
    for (const auto& delegate : delegates_) {
      if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) !=
          kTfLiteOk) {
        LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
      } else {
        LOG(INFO) << "Applied " << delegate.first << " delegate.";
      }
    }
    
    struct  timeval start;
    struct  timeval end;
    double diff;
    
    
    while (true) {
        cv::Mat image0;
        if( frames_index == frames - 1) {
            cam.set(cv::CAP_PROP_POS_FRAMES,0);
            frames_index = 0;
        }   
		
        frames_index++ ;
		
        auto success = cam.read(image0);
		
		if((!success) && (frames < 0)) {
            frames = frames_index-1;
			cam.set(cv::CAP_PROP_POS_FRAMES,0);
            frames_index = 0;
			auto success = cam.read(image0);
		}else if(!success) {
            std::cout << "Reading video fail" << std::endl;
            break;
        }
        
        auto cam_width = image0.cols;
        auto cam_height = image0.rows;
        cv::Mat image;
        resize(image0, image, Size(300,300));
        interpreter->AllocateTensors();
        
        auto inputs = interpreter->typed_input_tensor<uchar>(0);
        
        // feed input
        auto image_height=image.rows;
        auto image_width=image.cols;
        auto image_channels=3;
        int number_of_pixels = image_height * image_width * image_channels;
        int base_index = 0;
        // copy image to input as input tensor
        memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());
        
        interpreter->SetAllowFp16PrecisionForFp32(true);
        
        interpreter->SetNumThreads(1);
        
        gettimeofday(&start,NULL);
        interpreter->Invoke();
        gettimeofday(&end,NULL);
        
        diff = 1000 * (end.tv_sec-start.tv_sec)+ (end.tv_usec-start.tv_usec) / 1000;
        
        output_locations = interpreter->tensor(interpreter->outputs()[0]);
        auto output_data = output_locations->data.f;
        std::vector<float> locations;
        std::vector<float> cls;
        
        output_classes = interpreter->tensor(interpreter->outputs()[1]);
        auto out_cls = output_classes->data.f;
        num_detections   = interpreter->tensor(interpreter->outputs()[2]);
        auto nums = num_detections->data.f;
        for (int i = 0; i < 40; i++){
            auto output = output_data[i];
            locations.push_back(output);
            cls.push_back(out_cls[i]);
        }
        
        int count=0;
        std::vector<Object> objects;
        
        for(int j = 0; j <locations.size(); j+=4){
            auto ymin=locations[j]*cam_height;
            auto xmin=locations[j+1]*cam_width;
            auto ymax=locations[j+2]*cam_height;
            auto xmax=locations[j+3]*cam_width;
            auto width= xmax - xmin;
            auto height= ymax - ymin;
            
            float score = expit(nums[count]); 
            
            // auto id=outputClasses;
            Object object;
            object.class_id = cls[count];
            object.rec.x = xmin;
            object.rec.y = ymin;
            object.rec.width = width;
            object.rec.height = height;
            object.prob = score;
            objects.push_back(object);
            
            count+=1;
            
        }
        
        nms(objects,0.4);
        RNG rng(12345);
        for(int l = 0; l < objects.size(); l++)
        {
            Object object = objects.at(l);
            auto score=object.prob;
            if (score > 0.60f){
                Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
                auto cls = object.class_id;
                    
                cv::rectangle(image0, object.rec,color, 3);
                std::stringstream ss;
                ss << std::setprecision(2) << object.prob * 100;
                std::string cls_info = labels[cls+1] + ": " + ss.str()  + "%";
                cv::putText(image0, labels[cls+1], cv::Point(object.rec.x, object.rec.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(255, 100, 100), 2);

		if(s.accel == 1){
		    std::string text = "GPU Inference Time: " + std::to_string((int)diff) + " ms";
		    cv::putText(image0, text, cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(255, 100, 100), 2);
		}else if(s.accel == 2){
		    std::string text = "NPU Inference Time: " + std::to_string((int)diff) + " ms";
		    cv::putText(image0, text, cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(255, 100, 100), 2);
		}
		else{
		    std::string text = "CPUx1 Inference Time: " + std::to_string((int)diff) + " ms";
		    cv::putText(image0, text, cv::Point(10, 30),
		            cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(255, 100, 100), 2);
		}
	    }
        }
        resize(image0, image0, Size(640,480));
        if(s.accel == 1){
            cv::imwrite("/tmp/gpu.jpg", image0);
        }else if(s.accel == 2){
            cv::imwrite("/tmp/npu.jpg", image0);
        }
        else{
            cv::imwrite("/tmp/cpu.jpg", image0);
        }
    }
}

int main(int argc, char** argv) {
  
  int c;
  while (1)
  {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "a:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c)
    {
    case 'a':
      s.accel = strtol(optarg, nullptr, 10);
      break;
    default:
      exit(-1);
    }
  }

  run_inference();
  return 0;
}
