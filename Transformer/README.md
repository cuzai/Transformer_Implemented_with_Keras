# I. Transformerdd
## 1. Original architecture
![image](https://user-images.githubusercontent.com/13309017/215915484-9ae8e9b1-b3ea-4cfd-a44e-fe6d4d34a1db.png)<br><br><br><br>


## II. Code structure
### 1. Overall structure (Each is `tf.keras.layers.Layer`)
![image](https://user-images.githubusercontent.com/13309017/215958758-f43271ed-f49c-42f4-9ce3-f68875a91cb4.png)<br><br>

### 2. Positional Encoder
$PE(pos, 2i) = sin(pos/10000^{2i / d_{model}})$  
$PE(pos, 2i+1) = cos(pos/10000^{2i / d_{model}})$  
<img src=https://user-images.githubusercontent.com/13309017/215958946-881601df-bd0f-4950-ac2a-92a9ff2efe49.png width=350><br><br>

### 3. Encoder
<img src=https://user-images.githubusercontent.com/13309017/215959043-e23bb14c-c5d0-41bb-8e44-6ecd73fe3c53.png width=350><br><br>

### 4. Multi-head attention
$Scaled\text{ }dot\text{ }product\text{ }attention = softmax(Q_K^T / \sqrt{d_K})V$  
<img src=https://user-images.githubusercontent.com/13309017/215962559-21d5c5fb-614e-481e-b20c-6feb9e6c1633.png width=350> 
