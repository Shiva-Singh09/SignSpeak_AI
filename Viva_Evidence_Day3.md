# Viva Voce Evidence Report

## Date: Day 3

### Questions & Answers

1. **How does the sign language recognition model work?**
   - The model uses a CNN-LSTM architecture that processes hand gesture images through a convolutional base and sequences the features with LSTM for temporal patterns.

2. **What preprocessing steps are applied?**
   - Hand region detection using skin color segmentation
   - Resizing to 64x64 pixels
   - Normalization and data augmentation (rotation, scaling)

3. **Challenges faced during development**
   - Handling variable lighting conditions
   - Managing background clutter
   - Ensuring consistency across different users

4. **Future improvements**
   - Incorporating depth data from RGB-D cameras
   - Transfer learning with pre-trained models
   - Real-time inference optimization

### Sign-off
- Model architecture documented
- Preprocessing pipeline verified
- Test results recorded
- All questions answered satisfactorily