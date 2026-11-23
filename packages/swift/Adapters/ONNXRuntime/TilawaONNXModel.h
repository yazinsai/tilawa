// SPDX-License-Identifier: MIT
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN
/// Minimal Objective-C++ boundary around ONNX Runtime's C++ API. Add this header to the host target’s bridging header; the engine lives in the Tilawa Swift package.
@interface TilawaONNXModel : NSObject
- (nullable instancetype)initWithPath:(NSString *)path manifest:(NSData *)manifest error:(NSError **)error;
- (nullable NSData *)runFeatures:(NSData *)features error:(NSError **)error;
- (void)reset;
@end
NS_ASSUME_NONNULL_END
