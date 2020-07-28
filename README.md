# TVS
vision system

# yolo_tracker
Tracking objects in a video stream using yolo and darknet.  

Реализован:  
- [x] SORT.  
- [x] DeepSORT  

Получить darknet от AlexeyAB: [Darknet](https://github.com/AlexeyAB/darknet.git)

Компилировать darknet: [How to compile on Linux (using `make`)](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make)

Компилировать darknet с libdarknet.so (для использования в python): [How to use Yolo as DLL and SO libraries](https://github.com/AlexeyAB/darknet#how-to-use-yolo-as-dll-and-so-libraries) :

>on Linux
> - using `build.sh` or
> - build `darknet` using `cmake` or
> - set `LIBSO=1` in the `Makefile` and do `make`
