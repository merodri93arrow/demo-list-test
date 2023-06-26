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
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"

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
using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;

struct Object{
    cv::Rect rec;
    int      class_id;
    float    prob;
};
/*struct Settings {
  int accel = 0;
};*/
struct Settings {
  bool verbose = false;
  bool accel = false;
  TfLiteType input_type = kTfLiteFloat32;
  bool profiling = false;
  bool allow_fp16 = false;
  bool gl_backend = false;
  bool hexagon_delegate = false;
  bool xnnpack_delegate = false;
  int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  //string model_name = "./mobilenet_v1_1.0_224_quant.tflite";
  tflite::FlatBufferModel* model;
  //string input_bmp_name = "./grace_hopper.bmp";
  //string labels_file_name = "./labels.txt";
  int number_of_threads = 4;
  int number_of_results = 5;
  int max_profiling_buffer_entries = 1024;
  int number_of_warmup_runs = 2;
  int unit = 0;
};

Settings s;

float expit(float x) {
    return 1.f / (1.f + expf(-x));
}

class DelegateProviders {
 public:
  DelegateProviders() : delegate_list_util_(&params_) {
    delegate_list_util_.AddAllDelegateParams();
    delegate_list_util_.AppendCmdlineFlags(&flags_);

    // Remove the "help" flag to avoid printing "--help=false"
    params_.RemoveParam("help");
    delegate_list_util_.RemoveCmdlineFlag(flags_, "help");
  }

  // Initialize delegate-related parameters from parsing command line arguments,
  // and remove the matching arguments from (*argc, argv). Returns true if all
  // recognized arg values are parsed correctly.
  bool InitFromCmdlineArgs(int* argc, const char** argv) {
    // Note if '--help' is in argv, the Flags::Parse return false,
    // see the return expression in Flags::Parse.
    return tflite::Flags::Parse(argc, argv, flags_);
  }

  // According to passed-in settings `s`, this function sets corresponding
  // parameters that are defined by various delegate execution providers. See
  // lite/tools/delegates/README.md for the full list of parameters defined.
  void MergeSettingsIntoParams(const Settings& s) {
    // Parse settings related to GPU delegate.
    // Note that GPU delegate does support OpenCL. 'gl_backend' was introduced
    // when the GPU delegate only supports OpenGL. Therefore, we consider
    // setting 'gl_backend' to true means using the GPU delegate.
    if (s.gl_backend) {
      if (!params_.HasParam("use_gpu")) {
        LOG(WARN) << "GPU deleate execution provider isn't linked or GPU "
                     "delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_gpu", true);
        // The parameter "gpu_inference_for_sustained_speed" isn't available for
        // iOS devices.
        if (params_.HasParam("gpu_inference_for_sustained_speed")) {
          params_.Set<bool>("gpu_inference_for_sustained_speed", true);
        }
        params_.Set<bool>("gpu_precision_loss_allowed", s.allow_fp16);
      }
    }

    // Parse settings related to NNAPI delegate.
    if (s.accel) {
      if (!params_.HasParam("use_nnapi")) {
        LOG(WARN) << "NNAPI deleate execution provider isn't linked or NNAPI "
                     "delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_nnapi", true);
        params_.Set<bool>("nnapi_allow_fp16", s.allow_fp16);
      }
    }

    // Parse settings related to Hexagon delegate.
    if (s.hexagon_delegate) {
      if (!params_.HasParam("use_hexagon")) {
        LOG(WARN) << "Hexagon deleate execution provider isn't linked or "
                     "Hexagon delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_hexagon", true);
        params_.Set<bool>("hexagon_profiling", s.profiling);
      }
    }

    // Parse settings related to XNNPACK delegate.
    if (s.xnnpack_delegate) {
      if (!params_.HasParam("use_xnnpack")) {
        LOG(WARN) << "XNNPACK deleate execution provider isn't linked or "
                     "XNNPACK delegate isn't supported on the platform!";
      } else {
        params_.Set<bool>("use_xnnpack", true);
        params_.Set<bool>("num_threads", s.number_of_threads);
      }
    }
  }

  // Create a list of TfLite delegates based on what have been initialized (i.e.
  // 'params_').
  std::vector<ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates()
      const {
    return delegate_list_util_.CreateAllRankedDelegates();
  }

  std::string GetHelpMessage(const std::string& cmdline) const {
    return tflite::Flags::Usage(cmdline, flags_);
  }

 private:
  // Contain delegate-related parameters that are initialized from command-line
  // flags.
  tflite::tools::ToolParams params_;

  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;

  // Contains valid flags
  std::vector<tflite::Flag> flags_;
};


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

void RunInference(Settings* settings,
                  const DelegateProviders& delegate_providers) {
 
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("/home/root/.cache/eiq/eIQVideoSwitchCore2/model/ssd_mobilenet_v1_1_default_1.tflite");
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    // Resize input tensors, if desired.
    TfLiteTensor* output_locations = nullptr;
    TfLiteTensor* output_classes = nullptr;
    TfLiteTensor* num_detections = nullptr;
    // TfLiteTensor* scores = nullptr;
/*    int cam_id = 0;
    printf("Please input camera id:\n");
    std::cin >> cam_id;
    auto cam = cv::VideoCapture(cam_id);
*/
    int frames = 0;
    int frames_index = 0;

    auto cam = cv::VideoCapture("/home/root/.cache/eiq/eIQVideoSwitchCore2/media/video_device.mp4");
    frames = cam.get(cv::CAP_PROP_FRAME_COUNT);
	
    std::vector<std::string> labels;
    
    auto file_name="/home/root/.cache/eiq/eIQVideoSwitchCore2/model/labels.txt";
    std::ifstream input( file_name );
    for( std::string line; getline( input, line ); )
    {
        labels.push_back( line);
    }
    
    //interpreter->UseNNAPI(s.accel);
	
    auto delegates_ = delegate_providers.CreateAllDelegates();
    for (auto& delegate : delegates_) {
      const auto delegate_name = delegate.provider->GetName();
      if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) != kTfLiteOk) {
        LOG(ERROR) << "Failed to apply " << delegate_name << " delegate.";
        exit(-1);
      } else {
        LOG(INFO) << "Applied " << delegate_name << " delegate.";
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
        /*uchar *data_ptr = image.data;
        for (int i = 0; i < image.total() * image.elemSize(); i++) {
          inputs[i] = data_ptr[i];
        }*/
        
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
		//std::stringstream ss;
                //ss << std::setprecision(3) << diff;
		if(s.unit == 1){
		    std::string text = "GPU Inference Time: " + std::to_string((int)diff) + " ms";
		    cv::putText(image0, text, cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(255, 100, 100), 2);
		}else if(s.unit == 2){
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
        if(s.unit == 1){
            cv::imwrite("/tmp/gpu.jpg", image0);
        }else if(s.unit == 2){
            cv::imwrite("/tmp/npu.jpg", image0);
        }
        else{
            cv::imwrite("/tmp/cpu.jpg", image0);
        }
//      cv::imshow("cam", image0);
//      cv::waitKey(30);
    }
}

void display_usage(const DelegateProviders &delegate_providers) {
  LOG(INFO)
      << "\n"
      << delegate_providers.GetHelpMessage("mobilenet_ssd")
      << "\t--unit, -u [1:2]: 1: GPU, 2: NPU\n"
      << "\t--help, -h: Print this help message\n";
}

int main(int argc, char** argv) {
  
  int c;
  DelegateProviders delegate_providers;
  bool parse_result = delegate_providers.InitFromCmdlineArgs(
      &argc, const_cast<const char**>(argv));
  if (!parse_result) {
    display_usage(delegate_providers);
    return EXIT_FAILURE;
  }  
  
  while (1)
  {
    static struct option long_options[] = {
		{"unit_accelerated", required_argument, nullptr, 'u'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "u:h", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'u':
        s.unit = strtol(optarg, nullptr, 10);
		break;
	  case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage(delegate_providers);
        exit(-1);
      default:
        exit(-1);
    }
  }
  delegate_providers.MergeSettingsIntoParams(s);

  RunInference(&s, delegate_providers);
  return 0;
}
