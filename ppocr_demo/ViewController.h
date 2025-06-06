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

#import <UIKit/UIKit.h>
#import "onnxruntime_objc/onnxruntime.h"

@interface ViewController : UIViewController <UIImagePickerControllerDelegate, UINavigationControllerDelegate>

@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UILabel *lblTextReading;

@property (weak, nonatomic) IBOutlet UILabel *result;

@property (weak, nonatomic) IBOutlet UIButton *captureButton;
@property (weak, nonatomic) IBOutlet UIButton *btnScan;

@property (nonatomic, strong) ORTSession *sessionnn;
@property (nonatomic, strong) NSArray<NSString *> *imageList;
@property (nonatomic, strong) NSArray<NSString *> *characterChinese;
@property (nonatomic, assign) NSInteger currentIndex;
@property (nonatomic, assign) NSString *typePicture;
@property (nonatomic, strong) UIImagePickerController *imagePickerController;
@property (nonatomic, strong) UIImage *capturedImage;
@property (strong, nonatomic) UIImage *rotatedImage;

- (void)scanImage:(NSString *)imageString;
- (void)capturePhoto;
@end
