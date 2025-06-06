// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// clang-format off
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
// clang-format on
#import "ViewController.h"
#include "pipeline.h"
#include "timer.h"
#include <arm_neon.h>
#include <iostream>
#include <mutex>
#include <paddle_api.h>
#include <paddle_use_kernels.h>
#include <paddle_use_ops.h>
#include <string>
#include <regex>
#include <sstream>
#import <sys/timeb.h>
#include <vector>
#import <UIKit/UIKit.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import "ppocr_demo-Swift.h"

using namespace paddle::lite_api;
using namespace cv;

std::mutex mtx;
Pipeline *pipe_;
Timer tic;
long long count = 0;

@interface ViewController () <CvVideoCameraDelegate>
//@property(weak, nonatomic) IBOutlet UISwitch *flag_process;
//@property(weak, nonatomic) IBOutlet UISwitch *flag_video;
//@property(weak, nonatomic) IBOutlet UIImageView *preView;
//@property(weak, nonatomic) IBOutlet UISwitch *flag_back_cam;
@property(nonatomic, strong) CvVideoCamera *videoCamera;
@property(nonatomic, strong) UIImage *image;
@property(nonatomic) bool flag_init;
@property(nonatomic) bool flag_cap_photo;
@property(nonatomic) std::string dict_path;
@property(nonatomic) std::string config_path;
@property(nonatomic) cv::Mat cvimg;


@end

@implementation ViewController
@synthesize imageView;

- (void)viewDidLoad {
  [super viewDidLoad];
    
//    self.images = @[@"05076305080-2020-1-7.jpg", @"05076345352-2020-1-7.jpg"];
    
    self.imageList = @[];
    self.characterChinese = @[
        @"疗", @"绚", @"诚", @"娇", @"溜", @"题", @"贿", @"者", @"廖", @"更", @"纳", @"加", @"奉", @"公", @"一", @"就",
            @"汴", @"计", @"与", @"路", @"房", @"原", @"妇", @"-", @"其", @">", @":", @"]", @",",
            @"，", @"骑", @"刈", @"全", @"消", @"昏", @"傈", @"安", @"久", @"钟", @"嗅", @"不", @"影", @"处", @"驽", @"蜿",
            @"姿", @"掠", @"炳", @"泰", @"啧", @"脍", @"澜", @"汴" , @"上" , @"武"
    ];
    self.typePicture = @"1";
    self.currentIndex = 0;
    [self menuPicture];
    [self setupModel];
    [self.btnScan setHidden:YES];
//    [self initListImage];
//    [self.captureButton addTarget:self action:@selector(chooseOption) forControlEvents:UIControlEventTouchUpInside];
    
}

- (void) chooseOption {
    [self menuPicture];
}


- (void)capturePhoto{
    // Check if the device has a camera
    if ([UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera]) {
        // Initialize and configure UIImagePickerController
        self.imagePickerController = [[UIImagePickerController alloc] init];
        self.imagePickerController.delegate = self;
        self.imagePickerController.sourceType = UIImagePickerControllerSourceTypeCamera;
        self.imagePickerController.allowsEditing = YES;

        // Present the image picker
        [self presentViewController:self.imagePickerController animated:YES completion:nil];
    }
}

- (void)uploadImageFromLibrary {
    UIImagePickerController *imagePicker = [[UIImagePickerController alloc] init];
    imagePicker.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
    imagePicker.delegate = self;
    imagePicker.allowsEditing = YES; // Set to NO if you don't want editing.
    
    [self presentViewController:imagePicker animated:YES completion:nil];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<UIImagePickerControllerInfoKey, id> *)info {
    self.result.text = @"";
    // Get the captured image
    UIImage *capturedImage = info[UIImagePickerControllerEditedImage];
        if (!capturedImage) {
            capturedImage = info[UIImagePickerControllerOriginalImage];
        }
    
        self.capturedImage = capturedImage;

        // Dismiss the image picker
        [self dismissViewControllerAnimated:YES completion:^{
            // Call processImages with the captured image
            [self processImageCameraWithCapturedImage:capturedImage];
        }];
}

// This method is called when the user cancels the capture
- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {
    [self dismissViewControllerAnimated:YES completion:nil];
}



- (IBAction)rotateImage:(id)sender {
    if (!self.capturedImage) return;
        self.rotatedImage = [self rotateImageBy90Degrees:self.capturedImage];
        self.capturedImage = self.rotatedImage;
    self.result.text = @"";
    [self processImageCameraWithCapturedImage:self.capturedImage];
}

- (UIImage *)rotateImageBy90Degrees:(UIImage *)image {
    CGSize size = CGSizeMake(image.size.height, image.size.width); // Rotate dimensions
    UIGraphicsBeginImageContext(size);

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextTranslateCTM(context, size.width / 2, size.height / 2);
    CGContextRotateCTM(context, -M_PI_2); // 90 degrees in radians
    [image drawInRect:CGRectMake(-image.size.width / 2, -image.size.height / 2, image.size.width, image.size.height)];

    UIImage *rotatedImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();

    return rotatedImage;
}



- (void)scanImage:(NSString *)imageString {
    _image = [UIImage systemImageNamed:imageString];
    if (_image != nil) {
      printf("load image successed\n");
      imageView.image = _image;
    } else {
      printf("load image failed\n");
    }
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition =
        AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset =
        AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureVideoOrientation =
        AVCaptureVideoOrientationPortrait;
    self.videoCamera.rotateVideo = 90;
    self.videoCamera.defaultFPS = 30;
    [self.view insertSubview:self.imageView atIndex:0];
    self.cvimg.create(640, 480, CV_8UC3);
    NSString *path = [[NSBundle mainBundle] bundlePath];
    std::string paddle_dir = std::string([path UTF8String]);
    std::string det_model_file =
        paddle_dir + "/ch_ppocr_mobile_v2.0_det_slim_opt.nb";
    std::string rec_model_file =
        paddle_dir + "/ch_ppocr_mobile_v2.0_rec_slim_opt.nb";
    std::string cls_model_file =
        paddle_dir + "/ch_ppocr_mobile_v2.0_cls_slim_opt.nb";
    std::string img_path = paddle_dir + "/" + [imageString UTF8String];
    std::string output_img_path = paddle_dir + "/test_result.jpg";
    self.dict_path = paddle_dir + "/ppocr_keys_v1.txt";
    self.config_path = paddle_dir + "/config.txt";

    tic.start();
    cv::Mat srcimg = imread(img_path);
    pipe_ = new Pipeline(det_model_file, cls_model_file, rec_model_file,
                         "LITE_POWER_HIGH", 1, self.config_path, self.dict_path);
    std::ostringstream result;
    std::vector<std::string> res_txt;
    cv::Mat img_vis = pipe_->Process(srcimg, output_img_path, res_txt);
  
    tic.end();
    std::regex numericRegex("^[0-9]{3,}$");
    for (size_t i = 0; i < res_txt.size(); ++i) {
        
        NSString *initialString = [NSString stringWithUTF8String:res_txt[i].c_str()];
        NSString *resultString = [self removeChineseCharactersFromString:initialString];
        std::string targetStdString = [resultString UTF8String];
    
            if (std::regex_match(targetStdString, numericRegex)) {
                result << "Số đồng hồ nước là: " << res_txt[i] << "\n";
            } else {
                // Skip any non-numeric values
                std::cout << "Skipping non-numeric value: " << res_txt[i] << std::endl;
            }
        }
  
  
      self.result.numberOfLines = 10;
    
    NSString *resultString = [NSString stringWithUTF8String:result.str().c_str()];

    if ([resultString isEqual:@""]) {
        self.result.text = @"Không đọc được số nước";
    } else {
        self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
    }
      self.flag_init = true;
      self.imageView.image = MatToUIImage(img_vis);
}

- (NSString *)trimString:(NSString *)string basedOnType:(NSString *)type {
    if (string.length >= 5) {
        if ([type isEqualToString:@"1"]) {
            return [string substringToIndex:4];
        } else if ([type isEqualToString:@"2"]) {
            return [string substringToIndex:5];
        }
    }
    return string;
}

- (void)scanImagePicture:(UIImage *)capturedImage {
    
    if (capturedImage != nil) {
        self.imageView.image = capturedImage;
    } else {
        return;
    }
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.rotateVideo = 90;
    self.videoCamera.defaultFPS = 30;
    [self.view insertSubview:self.imageView atIndex:0];
    self.cvimg.create(640, 480, CV_8UC3);

    // Set up paths for models and configuration files
    NSString *path = [[NSBundle mainBundle] bundlePath];
    std::string paddle_dir = std::string([path UTF8String]);
    std::string det_model_file = paddle_dir + "/ch_ppocr_mobile_v2.0_det_slim_opt.nb";
    std::string rec_model_file = paddle_dir + "/ch_ppocr_mobile_v2.0_rec_slim_opt.nb";
    std::string cls_model_file = paddle_dir + "/ch_ppocr_mobile_v2.0_cls_slim_opt.nb";
    std::string output_img_path = paddle_dir + "/test_result.jpg";
    self.dict_path = paddle_dir + "/ppocr_keys_v1.txt";
    self.config_path = paddle_dir + "/config.txt";

    // Convert captured UIImage to cv::Mat
    cv::Mat srcimg;
    UIImageToMat(capturedImage, srcimg, true);
    cv::resize(srcimg, srcimg, cv::Size(1024, 760));
    cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);

    tic.start();
    pipe_ = new Pipeline(det_model_file, cls_model_file, rec_model_file,
                         "LITE_POWER_HIGH", 1, self.config_path, self.dict_path);
    std::ostringstream result;
    std::vector<std::string> res_txt;
    cv::Mat img_vis = pipe_->Process(srcimg, output_img_path, res_txt);

    tic.end();

    // Extract numeric values from OCR results
    std::regex numericRegex("^[0-9]{3,}$");
    for (size_t i = 0; i < res_txt.size(); ++i) {
        
        NSString *initialString = [NSString stringWithUTF8String:res_txt[i].c_str()];
        NSString *resultString = [self removeChineseCharactersFromString:initialString];
        std::string targetStdString = [resultString UTF8String];
    
        
            if (std::regex_match(targetStdString, numericRegex)) {
                resultString = [self trimString:resultString basedOnType:self.typePicture];
                targetStdString = [resultString UTF8String];
                result << "Số đồng hồ nước là: " << targetStdString << "\n";
            } else {
                // Skip any non-numeric values
                std::cout << "Skipping non-numeric value: " << targetStdString << std::endl;
            }
        }
    
//    for (size_t i = 0; i < res_txt.size(); ++i) {
//            std::string cleanedText = std::regex_replace(res_txt[i], chineseCharRegex, "");
//            if (std::regex_match(cleanedText, numericRegex)) {
//                result << "Số đồng hồ nước là: " << cleanedText << "\n";
//            } else {
//                // Bỏ qua bất kỳ giá trị không phải là số
//                std::cout << "Skipping non-numeric value: " << res_txt[i] << std::endl;
//            }
//        }

    // Update the result label
    self.result.numberOfLines = 10;
    NSString *resultString = [NSString stringWithUTF8String:result.str().c_str()];
    if ([resultString isEqualToString:@""]) {
        self.result.text = @"Không đọc được số nước";
    } else {
        self.result.text = resultString;
    }

    self.flag_init = true;
    self.imageView.image = MatToUIImage(img_vis);  // Display the processed image
}

- (NSString *)removeChineseCharactersFromString:(NSString *)initialString {
//            NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:@"[\\u4e00-\\u9fff]" options:0 error:nil];
//
//            NSString *cleanedString = [regex stringByReplacingMatchesInString:initialString options:0 range:NSMakeRange(0, [initialString length]) withTemplate:@""];
    
    
    NSMutableString *targetString = [initialString mutableCopy];
    
    // Iterate through each Chinese character in the array and replace it with an empty string
    for (NSString *chineseCharacter in self.characterChinese) {
        [targetString replaceOccurrencesOfString:chineseCharacter
                                      withString:@""
                                         options:NSLiteralSearch
                                           range:NSMakeRange(0, targetString.length)];
    }
    
    return [targetString copy];
}


//- (IBAction)nextImage_touch:(id)sender {
//    self.capturedImage = nil;
//    self.currentIndex++;
//    if (self.currentIndex >= self.imageList.count) {
//        self.currentIndex = 0;
//    }
//    [self processImagesWithImageString:self.imageList[self.currentIndex]];
//}

- (IBAction)scanImage_touch:(id)sender {
    if (self.capturedImage != nil) {
        [self scanImagePicture:self.capturedImage];
    } else {
        [self scanImage:self.imageList[self.currentIndex]];
    }
}

- (IBAction)uploadImage:(id)sender {
    [self uploadImageFromLibrary];
}




//- (IBAction)swith_video_photo:(UISwitch *)sender {
//  NSLog(@"%@", sender.isOn ? @"video ON" : @"video OFF");
//  if (sender.isOn) {
//    self.flag_video.on = YES;
//  } else {
//    self.flag_video.on = NO;
//  }
//}
//
//- (IBAction)cap_photo:(id)sender {
//  if (!self.flag_process.isOn) {
//    self.result.text = @"please turn on the camera firstly";
//  } else {
//    self.flag_cap_photo = true;
//  }
//}
//
//- (void)PSwitchValueChanged:(UISwitch *)sender {
//  NSLog(@"%@", sender.isOn ? @"process ON" : @"process OFF");
//  if (sender.isOn) {
//    [self.videoCamera start];
//  } else {
//    [self.videoCamera stop];
//  }
//}
//
//- (void)CSwitchValueChanged:(UISwitch *)sender {
//  NSLog(@"%@", sender.isOn ? @"back ON" : @"back OFF");
//  if (sender.isOn) {
//    if (self.flag_process.isOn) {
//      [self.videoCamera stop];
//    }
//    self.videoCamera.defaultAVCaptureDevicePosition =
//        AVCaptureDevicePositionBack;
//    if (self.flag_process.isOn) {
//      [self.videoCamera start];
//    }
//  } else {
//    if (self.flag_process.isOn) {
//      [self.videoCamera stop];
//    }
//    self.videoCamera.defaultAVCaptureDevicePosition =
//        AVCaptureDevicePositionFront;
//    if (self.flag_process.isOn) {
//      [self.videoCamera start];
//    }
//  }
//}

//- (void)processImage:(cv::Mat &)image {
//
//  dispatch_async(dispatch_get_main_queue(), ^{
//    if (self.flag_process.isOn) {
//      if (self.flag_init) {
//        if (self.flag_video.isOn || self.flag_cap_photo) {
//          self.flag_cap_photo = false;
//          if (image.channels() == 4) {
//            cvtColor(image, self->_cvimg, COLOR_RGBA2RGB);
//          }
//
//          tic.start();
//
//          std::vector<std::string> res_txt;
//          cv::Mat img_vis =
//              pipe_->Process(self->_cvimg, "output_img_result.jpg", res_txt);
//
//          tic.end();
//          // print recognized text
//          std::ostringstream result;
//          // print result
//          //    for (int i = 0; i < res_txt.size() / 2; i++) {
//          //        result << i << "\t" << res_txt[2*i] << "\t" <<
//          //        res_txt[2*i+1] << "\n";
//          //    }
//
//          result << "花费了" << tic.get_average_ms() << " ms\n";
//
//          cvtColor(img_vis, self->_cvimg, COLOR_RGB2BGR);
//          self.result.numberOfLines = 0;
//          self.result.text =
//              [NSString stringWithUTF8String:result.str().c_str()];
//          self.imageView.image = MatToUIImage(self->_cvimg);
//        }
//      }
//    }
//  });
//}

- (void)didReceiveMemoryWarning {
  [super didReceiveMemoryWarning];
  // Dispose of any resources that can be recreated.
}

@end
