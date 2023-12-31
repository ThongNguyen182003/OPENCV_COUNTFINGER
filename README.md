# Opencv + Firebase - COUNTFINGER

## Landmask and position
<p align="center">
  <img src="/image/HandTracking.png" alt="HandTracking"/>
</p>

## Installation

```sh
pip install firebase_admin
pip install mediapipe
```
### Set up Firebase

To use Firebase, please set it up using the following [link](https://www.freecodecamp.org/news/how-to-get-started-with-firebase-using-python/)

Access to <span style="color: red;">Project Settings</span> to get SDK
<p align="center">
  <img src="/image/ProjectSetting.png" alt="Project Settings"/>
</p>

Then set up code, example:
```sh
path_to_json = "C:/Users/thong/OneDrive/Máy tính/finger/source/firebase_admin-key.json"

cred_obj = firebase_admin.credentials.Certificate(path_to_json)
default_app = firebase_admin.initialize_app(cred_obj, {
    'databaseURL': 'https://opencv-numberfinger-default-rtdb.europe-west1.firebasedatabase.app/'
})
```

## Result
<p align="center">
  <img src="/image/realtimedtb.png" alt="result"/>
</p>