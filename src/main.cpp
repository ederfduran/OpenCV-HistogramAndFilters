#include "buttons.hpp"
#include <iostream>

// Use OpenCV command line parser Functions
const char* keys =
{
	"{help h usage ? | | print this message}"
    "{@image | | Image to process}"
};

void createButtons(){
    // Create UI buttons 
    cv::createButton("Show histogram", btn_cbk::showHistoCallback, NULL, cv::QT_PUSH_BUTTON, 0); 
    cv::createButton("Equalize histogram", btn_cbk::equalizeCallback, NULL, cv::QT_PUSH_BUTTON, 0); 
    cv::createButton("Lomography effect", btn_cbk::lomoCallback, NULL, cv::QT_PUSH_BUTTON, 0); 
    cv::createButton("Cartoonize effect", btn_cbk::cartoonCallback, NULL, cv::QT_PUSH_BUTTON, 0); 
}

int main(int argc , const char** argv){
    cv::CommandLineParser parser(argc,argv,keys);
    // description needed
    parser.about("Histogram ex.");
    if(parser.has("help")){
        parser.printMessage();
        return 0;
    }

    std::string img_file= parser.get<std::string>(0);
    //check if params are correctly parsed in his variables
    if(!parser.check()){
        parser.printErrors();
        return 0;
    }


    btn_cbk::img= cv::imread(img_file);
    if(btn_cbk::img.data == NULL){
        std::cout << "file image is missing"<< std::endl;
        return 0;
    }

    cv::namedWindow("Input");

    createButtons();

    cv::imshow("Input",btn_cbk::img);
    cv::waitKey(0);

    return 0;
}