[编译ROM]
## 下载
1. AOSP下载
[https://source.android.com/setup/build/downloading](https://source.android.com/setup/build/downloading)
#### 初始化 Repo 客户端
##### 安装 Repo 后，设置您的客户端以访问 Android 源代码代码库：
- 创建一个空目录来存放您的工作文件。如果您使用的是 MacOS，必须在区分大小写的文件系统中创建该目录。为其指定一个您喜欢的任意名称：
```
mkdir WORKING_DIRECTORY
cd WORKING_DIRECTORY
```
- 使用您的真实姓名和电子邮件地址配置 Git。要使用 Gerrit 代码审核工具，您需要一个与已注册的 Google 帐号关联的电子邮件地址。确保这是您可以接收邮件的有效地址。您在此处提供的姓名将显示在您提交的代码的提供方信息中。
```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
- 运行 repo init 以获取最新版本的 Repo 及其最近的所有错误更正内容。您必须为清单指定一个网址，该网址用于指定 Android 源代码中包含的各个代码库将位于工作目录中的什么位置。
```
curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
chmod a+x ~/bin/repo
repo init -u https://android.googlesource.com/platform/manifest
```
要检出“master”以外的分支，请使用 -b 指定相应分支。要查看分支列表，请参阅[源代码标记和编译版本](https://source.android.com/setup/start/build-numbers#source-code-tags-and-builds)。
```
repo init -u https://android.googlesource.com/platform/manifest -b android-8.1.0_r33
```
#### 下载 Android 源代码树
要将 Android 源代码树从默认清单中指定的代码库下载到工作目录，请运行以下命令：
```
repo sync
```
Android 源代码文件将位于工作目录中对应的项目名称下。初始同步操作将需要 1 个小时或更长时间才能完成。要详细了解 repo sync 和其他 Repo 命令，请参阅开发部分。

#### 下载Vendor Image
[https://developers.google.com/android/drivers#angler](https://developers.google.com/android/drivers#angler)
## 编译
```
source build/envsetup.sh
lunch aosp_angler-userdebug
make
```
## 烧录
```
#!/bin/sh
fastboot flash boot out/target/product/angler/boot.img
fastboot flash system out/target/product/angler/system.img
fastboot flash userdata out/target/product/angler/userdata.img
fastboot flash recovery out/target/product/angler/recovery.img
fastboot flash vendor out/target/product/angler/vendor.img
```


