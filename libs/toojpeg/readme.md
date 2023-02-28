# A JPEG encoder in a single C++ file

*This is a mirror of my library hosted at* https://create.stephan-brumme.com/toojpeg/

TooJpeg is a compact baseline JPEG/JFIF writer, written in C++11 (but looks like C for the most part).  
Its interface has only one function: `writeJpeg()` - and that's it !

My library supports the most common JPEG output color spaces:
- YCbCr444,
- YCbCr420 (=2x2 downsampled) and
- Y (grayscale)

Saving NASA's huge 21600x10800px [blue marble image](https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_21600.tif)
with quality=90 takes just 2.4 (YCbCr420) or 3.6 seconds (YCbCr444) on a standard x64 machine - faster than other small JPEG encoders.  
The compiled library enlarges your binary by about 7kb (CLang x64) or 12kb (GCC x64) in -O3 mode.  
Far more details can be found on the project homepage: https://create.stephan-brumme.com/toojpeg/

# How to use

1. create an image with any content you like, e.g. 1024x768, RGB (3 bytes per pixel)

```cpp
   auto pixels = new unsigned char[1024*768*3];
```

2. define a callback that receives the compressed data byte-by-byte 

```cpp
// for example, write to disk (could be anything else, too, such as network transfer, in-memory storage, etc.)
void myOutput(unsigned char oneByte) { fputc(oneByte, myFileHandle); }
```

3. start JPEG compression

```cpp
TooJpeg::writeJpeg(myOutput, mypixels, 1024, 768);
// actually there are some optional parameters, too
//bool ok = TooJpeg::writeJpeg(myOutput, pixels, width, height, isRGB, quality, downSample, comment);
```
