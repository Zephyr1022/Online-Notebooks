### Assignment 1 

Here is the course GitHub:

[https://github.com/utsabigdata/adv-ai](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2Futsabigdata%2Fadv-ai&data=02|01||50c4bdd046c245a110dc08d868bfb148|3a228dfbc64744cb88357b20617fc906|0|0|637374519803754587&sdata=rDS2n44hi%2Fsp%2FuHQYe1sxw5WTrw4clNe2UJaxo5IMp0%3D&reserved=0)

Regards,

Paul

Dear Ph.D. students,

 **There is no class on Sept 7 due to Labor Day.**

 I would like for everyone to audit the following **Coursera** class “**Neural Networks and Deep Learning**” By**Andrew Ng**. 

- Here is the link: [https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.coursera.org%2Flearn%2Fneural-networks-deep-learning%3Fspecialization%3Ddeep-learning&data=02|01||f3045737d6304070daa108d852aa21e2|3a228dfbc64744cb88357b20617fc906|0|0|637350237965606513&sdata=Svqugh59urwA7YmTaJ7PlBY4V3ErmPg3DEs7p%2F0Vvhc%3D&reserved=0), auditing the course with give you access to video lessons for free or you can listen to CS229 - Machine Learning (Stanford) by Andrew Ng. [https://see.stanford.edu/Course/CS229](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fsee.stanford.edu%2FCourse%2FCS229&data=02|01||f3045737d6304070daa108d852aa21e2|3a228dfbc64744cb88357b20617fc906|0|0|637350237965616501&sdata=ELLUyrXyNlr2RwxtxaLzLyig9iWpI62UP%2FenFUbGo4A%3D&reserved=0).

- **This is an undergraduate level course but helps you to review the content before our next class.**

- **Please review week 1-4 videos by next class Sept 14th.**

Regards,

Paul



Hi Everyone,

 

### Assignment #2 is up and available in

https://github.com/arundasan91/adv-ai-ml-research/tree/master/Assignment-2

You can just git pull in the cloud. I am also attaching the necessary files here as well.

When we were doing MLP, we found that the embeddings hold a lot of information. Similarly, we discussed that the convolution kernels are the trainable parameters of Conv operation. If we provide an input to the CNN (forward pass), the output of the multiplication operation with respect to each kernel is called an activation map (map since it is a 2D plane). This is similar to the embedding we found out for the DNN case. We can easily visualize activation maps in PyTorch by doing a forward pass on the model with a data input. I’m including a notebook to see how you can extract the weights of these kernels as well as the activations.

Here are the tasks:

- **Visualize the activation maps of each convolution layers**.

- - I’ve already done the visualization for Conv1. You have to do it for Conv2. However, you might want to remember that we apply MaxPool after Conv1. So, similarly, you have to apply MaxPool to the output of Conv1 before you try to get the activations of Conv2.
  - Pay attention to the shape of each output we get from Conv and MaxPool operations. I have added a model architecture in the Assignment to aid you with this.

- **Study the inter and intra class distances of embeddings extracted from two samples of numbers 6 and 7.**

- -  I’ve already written the code to extract numbers 6 and 7 as two variables. Get the embeddings (fc2 layer) from the CNN model for these four numbers (two each from 6 and 7). Now you have two embeddings for 6 and two for 7. Now, study the inter and intra class distance of these embeddings. You can use MSE and CrossEntropy to study the distances..

**Pleas**e CC Dr. Rad in future correspondences.

Thank you,

Arun



I uploaded the slides: 

[https://github.com/utsabigdata/adv-ai](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2Futsabigdata%2Fadv-ai&data=04|01||6dda345e58714508486608d876a7966c|3a228dfbc64744cb88357b20617fc906|0|0|637389809444496581|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|1000&sdata=UjfJuFE6cUbenXEMYwis03Mhe04oLg1o6Rdj22DksrI%3D&reserved=0)



### **Assignment 3**: visualizing and understanding Convolution Networks: Implement this paper in Pytorch 

**Due Date: 10/19/2020**

Please read the following Paper by Zeiler and Fergus (NYU)

[https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf](https://nam11.safelinks.protection.outlook.com/?url=https:%2F%2Fcs.nyu.edu%2F~fergus%2Fpapers%2FzeilerECCV2014.pdf&data=04|01||6dda345e58714508486608d876a7966c|3a228dfbc64744cb88357b20617fc906|0|0|637389809444496581|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|1000&sdata=RNkngoB50K4M%2FDVbqEDBOGJjj%2BBc1TuSDE1SITewS0M%3D&reserved=0)

Here is Matt Zeiler’s presentation

[https://www.youtube.com/watch?v=ghEmQSxT6tw](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DghEmQSxT6tw&data=04|01||6dda345e58714508486608d876a7966c|3a228dfbc64744cb88357b20617fc906|0|0|637389809444506573|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|1000&sdata=%2FwHzQgc5rCTMhPpo%2FdaUgbjqIkrRh43bV2XmGTGkc3o%3D&reserved=0)

Code in Tensorflow: [https://github.com/FHainzl/Visualizing_Understanding_CNN_Implementation](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2FFHainzl%2FVisualizing_Understanding_CNN_Implementation&data=04|01||6dda345e58714508486608d876a7966c|3a228dfbc64744cb88357b20617fc906|0|0|637389809444506573|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|1000&sdata=SWlD9AsvjMLbGdS04ZV61z4BcRrZ55DMntAq9cRHRcM%3D&reserved=0)

Regards,

Paul



### Assignment 4: IS-7033-001-Fall-2020-Topics: ML Research (Due Date Oct, 27)

Implement Convolution Autoencoder using the following model:

[https://www.kaggle.com/ljlbarbosa/convolution-autoencoder-pytorch/notebook](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.kaggle.com%2Fljlbarbosa%2Fconvolution-autoencoder-pytorch%2Fnotebook&data=04|01||6dda345e58714508486608d876a7966c|3a228dfbc64744cb88357b20617fc906|0|0|637389809444476597|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|1000&sdata=sVvPQ9QIqB1%2B2mOCjI2rbyO2JHVNX2fHgSPh%2F%2FGeUfA%3D&reserved=0)



Read this sample white paper on Convolution Autoencoder:

[http://users.cecs.anu.edu.au/~Tom.Gedeon/conf/ABCs2018/paper/ABCs2018_paper_58.pdf](https://nam11.safelinks.protection.outlook.com/?url=http:%2F%2Fusers.cecs.anu.edu.au%2F~Tom.Gedeon%2Fconf%2FABCs2018%2Fpaper%2FABCs2018_paper_58.pdf&data=04|01||6dda345e58714508486608d876a7966c|3a228dfbc64744cb88357b20617fc906|0|0|637389809444486591|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|1000&sdata=YWXt8GcKLrNNsuS39A3SGat5zTbe6ZcxVPlqQYrYmnQ%3D&reserved=0)

 

After training (feel free to share the save training models with each other)

1. save the kernels for each layer
2. for a sample input (airplane) visualize the activation maps for each later (decoder and encoder).
3. Write 2-3 page paper summarizing your analysis

Regards,

Paul



###### #research 

https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
https://bastings.github.io/annotated_encoder_decoder/   
https://analyticsindiamag.com/hands-on-guide-to-implement-deep-autoencoder-in-pytorch-for-image-reconstruction/    



### Assignment 5: IS-7033-001-Fall-2020-Topics: ML Research (Due Date Nov. 2nd)

Implement Adversarial Example (FGSM) using the following code:

[https://pytorch.org/tutorials/beginner/fgsm_tutorial.html](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fpytorch.org%2Ftutorials%2Fbeginner%2Ffgsm_tutorial.html&data=04|01||360f9820c00f4c6091f408d87ba8e290|3a228dfbc64744cb88357b20617fc906|0|0|637395312580340615|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|1000&sdata=Ipm5%2FHKM1xWGwk%2FFAR4u0M85HE9SmBJ8lFx8BCUIh7U%3D&reserved=0)



Read this paper on “EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES”

[https://arxiv.org/pdf/1412.6572.pdf](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Farxiv.org%2Fpdf%2F1412.6572.pdf&data=04|01||360f9820c00f4c6091f408d87ba8e290|3a228dfbc64744cb88357b20617fc906|0|0|637395312580350610|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|1000&sdata=1G3dHIJ5RIxwfhpDszSrtv4OBOGI48G3PQIKQvJNNaU%3D&reserved=0) 

After generating sample attacks:

1. Compare the attack noise for each data sample
2. Is the attack for all data samples the same (universal)

 **Bonus:** 

I uploaded a code for training using VGG training using CIFAR [Vgg16-CIFAR.ipynb](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2Futsabigdata%2Fadv-ai%2Fblob%2Fmaster%2Fcode%2FVgg16-CIFAR.ipynb&data=04|01||360f9820c00f4c6091f408d87ba8e290|3a228dfbc64744cb88357b20617fc906|0|0|637395312580350610|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|1000&sdata=fHhQ2EkxGqI96fEh0noq8GOoHmTqJiOK3XGAddWj9CY%3D&reserved=0)

[https://github.com/utsabigdata/adv-ai/tree/master/code](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2Futsabigdata%2Fadv-ai%2Ftree%2Fmaster%2Fcode&data=04|01||360f9820c00f4c6091f408d87ba8e290|3a228dfbc64744cb88357b20617fc906|0|0|637395312580360605|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|1000&sdata=rqOxwJA04Ak7dsVqdgB2yL8dSOBt976IARkZs%2BfRzAc%3D&reserved=0)

After you trained the model, build FGSM attack for some of the sample test data and compare the new attacks with the previous attacks  

 Regards,

Paul

