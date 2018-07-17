```
系统环境：
Ubuntu 18.04
```
##下载anbox项目代码
```
git clone https://github.com/anbox/anbox.git
```
##制作android.img
下载anbox android系统源码
```
mkdir $HOME/anbox-work
cd $HOME/anbox-work
repo init -u https://github.com/anbox/platform_manifests.git -b anbox
repo sync -j4
```
编译
```
. build/envsetup.sh
lunch anbox_x86_64-userdebug
make -j4

The complete list of supported build targets:
anbox_x86_64-userdebug
anbox_armv7a_neon-userdebug
anbox_arm64-userdebug
```
制作android.img
```
cd $HOME/anbox-work/anbox
scripts/create-package.sh \
    $PWD/../out/target/product/x86_64/ramdisk.img \
    $PWD/../out/target/product/x86_64/system.img
```