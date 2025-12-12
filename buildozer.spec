[app]
title = YouTube Partner Estimator
package.name = youtubepartnerestimator
package.domain = org.example

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 0.1
requirements = python3,kivy==2.2.1,google-api-python-client,transformers,torch,numpy,isodate,regex

[buildozer]
log_level = 2

[app]
android.permissions = INTERNET, ACCESS_NETWORK_STATE
android.api = 30
android.minapi = 21
android.ndk = 23b
android.sdk = 30

# If you want to use buildozer to build for Android
# buildozer init will create the default buildozer.spec file
# Then you can run buildozer android debug to build the APK