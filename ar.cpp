#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <memory>

using namespace cv;
using namespace std;

const GLdouble size=0.5;
const Scalar color = (0, 255, 0);

class AR_GL_CV_Tester
{
public:
  AR_GL_CV_Tester();
  AR_GL_CV_Tester(const AR_GL_CV_Tester& other);
  ~AR_GL_CV_Tester();

  bool initGL(); //setup GL for drawing
  bool initCV(); //setup CV
  void display(); //our callback function
  void drawQuad(Mat&, Mat, Scalar);
  void handleMarkerTracking();
  void captureProcessWebcam();
  const Mat& getImg() const {return m_img;}

private:
  Mat m_img;
  Mat m_rvec;
  Mat m_tvec;
  Mat m_intrinsics;
  Mat m_distortion;
  Mat m_glViewMatrix;
  VideoCapture m_capture;
  vector<vector<Point> > m_contours;
  vector<Mat> m_squares;
	GLuint textureID;
  GLuint textureID2;

};

AR_GL_CV_Tester::AR_GL_CV_Tester()
{

}

AR_GL_CV_Tester::AR_GL_CV_Tester(const AR_GL_CV_Tester& other)
{

}

AR_GL_CV_Tester::~AR_GL_CV_Tester()
{

}


void AR_GL_CV_Tester::drawQuad(Mat& image, Mat points, Scalar color) {
    cout << points.at<Point2f>(0,0) << " " << points.at<Point2f>(0,1) << " " << points.at<Point2f>(0,2) << " " << points.at<Point2f>(0,3) << endl;
    line(image, points.at<Point2f>(0,0), points.at<Point2f>(0,1), color);
    line(image, points.at<Point2f>(0,1), points.at<Point2f>(0,2), color);
    line(image, points.at<Point2f>(0,2), points.at<Point2f>(0,3), color);
    line(image, points.at<Point2f>(0,3), points.at<Point2f>(0,0), color);
}

//INIT TO GL
bool AR_GL_CV_Tester::initGL(void)
{
    //select clearing (background) color
    glClearColor(1.0, 1.0, 1.0, 1.0);

    //initialize viewing values
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);

    //enable textures
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &textureID); //videobackground
    glGenTextures(1, &textureID2); //teapot texture


    return true;
}

bool AR_GL_CV_Tester::initCV()
{
  //read in the camera calibration
  FileStorage fs("../../calibrate/build/out_camera_data.xml", FileStorage::READ);
  fs["Camera_Matrix"] >> m_intrinsics;
  fs["Distortion_Coefficients"] >> m_distortion;
  if (m_intrinsics.rows != 3 || m_intrinsics.cols != 3 || m_distortion.rows != 5 || m_distortion.cols != 1) {
      cout << "Run calibration (in ../calibrate/) first!" << endl;
      return false;
  }
  //open the camera
  bool result = m_capture.open(-1);
  if (! result){
    return false;
  }
  return true;
}

//do all the OCV processing here
void AR_GL_CV_Tester::handleMarkerTracking()
{
    Scalar green(0, 255, 0);
  //clear containers
  m_contours.clear();
  m_squares.clear();
  //convert to greyscale
  Mat grayImage;
  cvtColor(m_img, grayImage, CV_RGB2GRAY);
  //blur
  Mat blurredImage;
  blur(grayImage, blurredImage, Size(5, 5));
  //threshold
  Mat threshImage;
  threshold(blurredImage, threshImage, 128.0, 255.0, THRESH_OTSU);
  //get the contours
  findContours(threshImage, m_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  for (auto contour : m_contours) {
      vector<Point> approx;
      approxPolyDP(contour, approx, arcLength(Mat(contour), true)*0.02, true);
      if( approx.size() == 4 &&
          fabs(contourArea(Mat(approx))) > 1000 &&
          isContourConvex(Mat(approx)) )
      {
          Mat squareMat;
          Mat(approx).convertTo(squareMat, CV_32FC3);
          m_squares.push_back(squareMat);
      }
  }

  if (m_squares.size() > 0) {
    vector<Point3f> objectPoints = {Point3f(-1, -1, 0), Point3f(-1, 1, 0), Point3f(1, 1, 0), Point3f(1, -1, 0)};
    Mat objectPointsMat(objectPoints);
    cout << "objectPointsMat: " << objectPointsMat.rows << ", " << objectPointsMat.cols << endl;
    cout << "squares[0]: " << m_squares[0] << endl;
    //Mat rvec;
    //Mat tvec;
    solvePnP(objectPointsMat, m_squares[0], m_intrinsics, m_distortion, m_rvec, m_tvec);

    cout << "rvec = " << m_rvec << endl;
    cout << "tvec = " << m_tvec << endl;

    /*
    Mat rotation, viewMatrix(4, 4, CV_64F);
    Rodrigues(m_rvec, rotation);
    for(unsigned int row=0; row<3; ++row)
    {
      for(unsigned int col=0; col<3; ++col)
      {
        viewMatrix.at<double>(row, col) = rotation.at<double>(row, col);
      }
      viewMatrix.at<double>(row, 3) = m_tvec.at<double>(row, 0);
    }

    viewMatrix.at<double>(3, 3) = 1.0f;
    m_glViewMatrix = Mat::zeros(4, 4, CV_64F);
    transpose(viewMatrix , m_glViewMatrix);
    */
    drawQuad(m_img, m_squares[0], green);
  }



  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR); // scale linearly when image bigger than texture
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); // scale linearly when image smalled than texture
  glTexImage2D(GL_TEXTURE_2D, 0, 3, m_img.cols, m_img.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, m_img.data);

}

void AR_GL_CV_Tester::captureProcessWebcam()
{

  if (m_capture.read(m_img))
  {
    cout << "read OK" << endl;
  }
  else
  {
    cout << "bad webcam read eh";
  }
}

//OUR DISPLAY CALLBACK FUNCTION
void AR_GL_CV_Tester::display(void)
{

    //Clear all pixels
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    //draw BKGRND
    glBegin(GL_QUADS);

    float x = 1;
    float y = 1;

    glTexCoord2f(0.0f, 0.0f); glVertex3f(-x, -y, 0.0f);
    glTexCoord2f(0.0f, 1.0f);glVertex3f( x, -y, 0.0f);
    glTexCoord2f(1.0f, 1.0f);glVertex3f( x, y, 0.0f);
    glTexCoord2f(1.0f, 0.0f);glVertex3f(-x, y, 0.0f);

    //draw cube
    glEnd();


    glPushMatrix();

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(&m_glViewMatrix.at<double>(0, 0));

    glBindTexture(GL_TEXTURE_2D, textureID2);
    glutSolidTeapot(size);

    // Don't wait start processing buffered OpenGL routines
    glFlush();

}

//global ptr and display callback wrapper
AR_GL_CV_Tester* g_ARGL;
void oglDraw()
{
  if (g_ARGL)
  {
    g_ARGL->display();
  }
}

void oglIdle()
{
  if (g_ARGL)
  {
    g_ARGL->captureProcessWebcam();
    g_ARGL->handleMarkerTracking();
    glutPostRedisplay();
  }
}

int main( int argc, char** argv ){

  //VideoCapture cap(-1);
  //Mat img;
  AR_GL_CV_Tester ar;
  g_ARGL = &ar;
  bool result;

  result = ar.initCV();
  if (!result)
  {
    return -1;
  }

  //namedWindow("theWindow", CV_WINDOW_AUTOSIZE);
  Mat test;
  //test = ar.getImg();
  while(1)
  {
    ar.captureProcessWebcam();
    ar.handleMarkerTracking();
    imshow("theWindow", ar.getImg() );
    if (waitKey(20) == 27) {return 0;}
  }

  /*
  glutInit(&argc, argv);
  //set display mode
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  //Set the window size
  glutInitWindowSize(250,250);
  //Set the window position
  glutInitWindowPosition(100,100);
  //Create the window
  glutCreateWindow("A Simple OpenGL Windows Application with GLUT");

  //Call init (initialise GLUT
  result = ar.initGL();
  if (!result)
  {
    return -1;
  }
  */
  //init CV - load calibration


  //glutDisplayFunc(oglDraw);
  //glutIdleFunc(oglIdle);
  //glutMainLoop();
/*
  Scalar red(255, 0, 0);
  Scalar green(0, 255, 0);
  Scalar blue(0, 0, 255);





//Initiate camera

VideoCapture cap(-1);
Mat img;

namedWindow("theWindow", CV_WINDOW_AUTOSIZE);

while(true){

//if can't read from camera, program dies

  if (!cap.read(img)) { return -1; }

  //convert to greyscale
  Mat grayImage;
  cvtColor(img, grayImage, CV_RGB2GRAY);
  //blur
  Mat blurredImage;
  blur(grayImage, blurredImage, Size(5, 5));
  //threshold
  Mat threshImage;
  threshold(blurredImage, threshImage, 128.0, 255.0, THRESH_OTSU);

  std::vector<std::vector<Point> > contours;
  findContours(threshImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  Scalar color(0, 255, 0);

  //fin dthe poly of the square
  std::vector<Mat> squares;
  for (auto contour : contours) {
      vector<Point> approx;
      approxPolyDP(contour, approx, arcLength(Mat(contour), true)*0.02, true);
      if( approx.size() == 4 &&
          fabs(contourArea(Mat(approx))) > 1000 &&
          isContourConvex(Mat(approx)) )
      {
          Mat squareMat;
          Mat(approx).convertTo(squareMat, CV_32FC3);
          squares.push_back(squareMat);
      }
  }

  if (squares.size() > 0) {
    vector<Point3f> objectPoints = {Point3f(-1, -1, 0), Point3f(-1, 1, 0), Point3f(1, 1, 0), Point3f(1, -1, 0)};
    Mat objectPointsMat(objectPoints);
    cout << "objectPointsMat: " << objectPointsMat.rows << ", " << objectPointsMat.cols << endl;
    cout << "squares[0]: " << squares[0] << endl;
    Mat rvec;
    Mat tvec;
    solvePnP(objectPointsMat, squares[0], intrinsics, distortion, rvec, tvec);

    cout << "rvec = " << rvec << endl;
    cout << "tvec = " << tvec << endl;

    drawQuad(img, squares[0], green);

    //draw a calibrated line on the marker
    vector<Point3f> line3d = {{0, 0, 0}, {0, 0, 1}};
    vector<Point2f> line2d;
    projectPoints(line3d, rvec, tvec, intrinsics, distortion, line2d);
    cout << "line2d = " << line2d << endl;
    line(img, line2d[0], line2d[1], red);
}

  //drawContours(img, squares, -1, color);
  imshow("theWindow", img );

  if (waitKey(20) == 27) {return 0;}

}
*/
return 0;
}
