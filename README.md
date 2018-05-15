# brick_squad
Submission for iMaterialist Challenge (Fashion) on Kaggle

## Using Azure server

1. Make sure you've clicked on the invitation link in your KU email.
2. Go to [portal.azure.com](http://portal.azure.com) and login with your KU credentials.
3. Once logged in, make sure you're in the right directory: click on your profile picture in the top right corner, then "Switch Directory". Choose the one starting with `christianigeloutlook...`.
4. Click on "All resources" in the left sidebar and select "blocksquad". In the new window, you can click on "Start" or "Stop" depending on what you want to do with the server. The public IP of the server is also there.
5. When the server is running, you can ssh into it. Open terminal and type `ssh blocksquad@<public ip>`, where "\<public ip\>" is the public IP address of the server. Use the password we posted on Facebook.
6. Always use the "lsda" virtual environment (otherwise some stuff may not work). Type: `source lsda/bin/activate`.
7. To run the jupyter notebook, type `./jupyter.sh`. The go to http://\<public ip\>:8888 and use the same password to login.
8. After you finished working on the server, stop it via Azure's dashboard.

## Links

1. [Training set of 10000 random unique training images on Google Drive](https://drive.google.com/file/d/1LB91lK6Ksk24nWAk4UxuHLT04UWl7lJ3/view?usp=sharing).
2. [ML-KNN: A lazy learning approach to multi-label learning
](https://drive.google.com/file/d/1XX1ezSDiqpJzVr9sNEU69zaRPzzs4TRW/view?usp=sharing)
3. [Choosing the right estimator (scikit-learn cheet-sheet)](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
4. [Image recognition with TensorFlow](https://www.tensorflow.org/tutorials/image_recognition)
