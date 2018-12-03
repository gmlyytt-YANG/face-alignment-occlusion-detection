# Task Description 
There are 3 steps for face alignment in the condition of occlusion. 
1. Rougn Face Alignment(DL)
2. Occlusion Detection(to judge whether each point is occluded)
3. Precise Face Alignment: Based on result of occlusion detection, we will just update the location of points which are not occluded. 

# Rough Face Alignment 

# Occlusion Detection 
  1. model structure
  
    Vgg16(cut off the second fully connected layer) + 68 binary output
  
  2. loss function 
    
    binary_crossentropy(I'm trying the mae, which could be more rational)
    
# Precise Face Alignment
