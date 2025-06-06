
//  extension.swift
//  ppocr_demo
//
//  Created by Tran Tan Duc on 6/11/24.
//  Copyright © 2024 Li,Xiaoyang(SYS). All rights reserved.


import UIKit
import onnxruntime_objc
import ObjectiveC

@objc extension ViewController  {
    
//    private struct AssociatedKeys {
//           static var sessionKey: Void?
//       }
//
//    @objc var images: [String]? {
//            get {
//                return objc_getAssociatedObject(self, &AssociatedKeys.sessionKey) as? [String]
//            }
//            set {
//                objc_setAssociatedObject(self, &AssociatedKeys.sessionKey, newValue, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
//            }
//        }
    
    
    
    @objc func menuPicture() {
            let dataSource = ["Chụp ảnh đồng hồ 4 số", "Chụp ảnh đồng hồ 5 số"]
            
            let actionClosure = { (action: UIAction) in
                if (action.title == "Chụp ảnh đồng hồ 4 số") {
                    self.typePicture = "1"
                    self.capturePhoto()
                } else {
                    self.typePicture = "2"
                    self.capturePhoto()
                }
                
                }
            var menuChildren: [UIMenuElement] = []
                for fruit in dataSource {
                    menuChildren.append(UIAction(title: fruit, handler: actionClosure))
                }
            
            self.captureButton.menu = UIMenu(options: .displayInline, children: menuChildren)
            self.captureButton.showsMenuAsPrimaryAction = true
            self.captureButton.changesSelectionAsPrimaryAction = true
        
    }
    
    @objc func initListImage() {
        
        let imagePaths = Bundle.main.paths(forResourcesOfType: nil, inDirectory: "images")
        self.imageList = imagePaths.map { URL(fileURLWithPath: $0).lastPathComponent }
        if (imageList.count > 0) {
            self.processImages(imageString: self.imageList[self.currentIndex])
        }
    }
    
     @objc func setupModel() {
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx") else {
            print("Model not found")
            return
        }
        do {
            let options = try ORTSessionOptions()
            sessionnn = try ORTSession(env: ORTEnv(loggingLevel: .warning), modelPath: modelPath, sessionOptions: options)
        } catch {
            print("Failed to create ORTSession: \(error)")
        }
    }
    
    @objc func processImages(imageString:String) {
        self.lblTextReading.text = ""
        self.result.text = ""
        let imageString: String = imageString;
        let components = imageString.split(separator: ".")
            let filename = String(components[0])
            let fileExtension = String(components[1])
            
            if let imagePath = Bundle.main.path(forResource: filename, ofType: fileExtension),
               let image = UIImage(contentsOfFile: imagePath) {
                imageView.image = image
                if let waterMeterReading = runModel(on: image) {
                    displayResult(reading: waterMeterReading)
                }
            } else {
                print("Image not found")
            }
        
    }
    
    @objc func processImageCamera(capturedImage: UIImage? = nil) {
        // Clear text fields before processing) {
        if let image = capturedImage {
                imageView.image = image
                if let waterMeterReading = runModel(on: image) {
                    displayResult(reading: waterMeterReading)
                }
                return
            }
    }
    
    @objc func runModel(on image: UIImage) -> String? {
        guard let session = sessionnn else {
            print("Model session is not available.")
            return nil
        }
        
        guard let pixelBuffer = image.pixelBuffer(width: 224, height: 224) else {
            print("Failed to convert image to pixel buffer.")
            return nil
        }
        
        guard let inputData = pixelBufferToData(pixelBuffer) else {
            print("Failed to convert pixel buffer to data.")
            return nil
        }
        
        do {
            
            let inputTensor = try ORTValue(tensorData: inputData, elementType: .float, shape: [1, 3, 224, 224])
            let outputs = try session.run(withInputs: [session.inputNames().first!: inputTensor], outputNames: [session.outputNames().first!], runOptions: ORTRunOptions())

            if let outputTensor = outputs[try session.outputNames().first!] {
                        let outputString = parseORTValue(outputTensor)
                return outputString
                    }
            return "aaa"
           
        } catch {
            print("Failed to run model: \(error)")
            return nil
        }
    }
    
    func parseORTValue(_ ortValue: ORTValue) -> String {
           do {
               // Get the tensor data
               let outputData = try ortValue.tensorData() as Data
               
               // Convert the data to a Swift array
               let outputArray = outputData.withUnsafeBytes {
                   Array(UnsafeBufferPointer<Float>(start: $0.bindMemory(to: Float.self).baseAddress!, count: outputData.count / MemoryLayout<Float>.size))
               }
               
               // Convert the output array to a string
               let outputString = outputArray.map { String($0) }.joined(separator: ", ")
               
               // Return the output string
               print(outputString)
               return "\(outputString)"
           } catch {
               return "Failed to parse ORTValue: \(error)"
           }
       }
    
    private func displayResult(reading: String) {
        
        
        
        let components = reading.split(separator: ",")
        if components.count == 3 {
            // Convert each component to Double
            let readingsArray = components.compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            if readingsArray.count == 3 {
                print("Conversion successful: \(readingsArray)")
                // Output: [0.14274558, 0.34216392, 0.5150905]
                processReadings(readings: readingsArray)
            }
        } else {
            self.lblTextReading.text = "Không tìm thấy đồng hồ nước trong ảnh"
        }
    }
    
    private func processReadings(readings: [Double]) {
        let threshHoldClockExist: Double = 0.229
        let threshHoldClockVisibility: Double = 0.3
        let threshHoldWaterUsageVisibility: Double = 0.36
        
        var clockExist = false
        var clockVisible = false
        var waterUsageVisible = false
        
        var clockExistEva: Double = 0.0
        var clockVisibleEva: Double = 0.0
        var waterUsageEva: Double = 0.0
        
        for (index, value) in readings.enumerated() {
            switch index {
            case 0:
                // Process clockExist
                if value >= threshHoldClockExist {
                    clockExist = true
                }
                clockExistEva = value
            case 1:
                // Process clockVisible
                if value >= threshHoldClockVisibility {
                    clockVisible = true
                }
                clockVisibleEva = value
            case 2:
                // Process waterUsageVisible
                if value >= threshHoldWaterUsageVisibility {
                    waterUsageVisible = true
                }
                waterUsageEva = value
            default:
                break
            }
        }
//        
//        // Display the results in the displayResult method
//        let result = """
//        Clock Exist: \(clockExist), Value: \(clockExistEva)
//        Clock Visible: \(clockVisible), Value: \(clockVisibleEva)
//        Water Usage Visible: \(waterUsageVisible), Value: \(waterUsageEva)
//        """
        
        if (waterUsageVisible) {
            self.lblTextReading.text = "Phát hiện đồng hồ nước trong ảnh, scan để đọc số đồng hồ nước"
            self.btnScan.isUserInteractionEnabled = true
            self.btnScan.isHidden = false
        } else {
            self.lblTextReading.text = "Không phát hiện đồng hồ nước trong ảnh"
            self.btnScan.isHidden = true
            self.btnScan.isUserInteractionEnabled = false
        }
        
    }
    
    private func pixelBufferToData(_ pixelBuffer: CVPixelBuffer) -> NSMutableData? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytePerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        let data = NSMutableData()
        
        for y in 0..<height {
            let row = baseAddress.advanced(by: y * bytePerRow)
            let rowData = Data(bytes: row, count: width * 4) // Assuming 4 bytes per pixel (RGBA)
            
            // Normalize pixel values to [0, 1] and extract RGB only
            rowData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                for x in stride(from: 0, to: rowData.count, by: 4) {
                    var r = Float(ptr[x]) / 255.0
                    var g = Float(ptr[x + 1]) / 255.0
                    var b = Float(ptr[x + 2]) / 255.0
                    data.append(&r, length: MemoryLayout<Float>.size)
                    data.append(&g, length: MemoryLayout<Float>.size)
                    data.append(&b, length: MemoryLayout<Float>.size)
                }
            }
        }
        
        return data
    }
    
}


extension UIImage {
    func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attributes as CFDictionary, &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        
        let context = CGContext(data: pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        guard let cgImage = self.cgImage else { return nil }
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, [])
        
        return buffer
    }
}

extension NSMutableData {
    func toArray<T>(type: T.Type) -> [T] {
        let count = self.length / MemoryLayout<T>.size
        var array = [T](repeating: 0 as! T, count: count)
        self.getBytes(&array, length: self.length)
        return array
    }
}
