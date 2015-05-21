//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

struct OtherOpt {
    OtherOpt() : deviceNumber(0), showHelp(false) {}

    std::string processName;
    int deviceNumber;
    bool showHelp;
    
    void setValidDeviceNumber( int i ) {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if( i < 0 || i > deviceCount ) {
            Catch::cout()<<"The device number is incorrect, please set valid cuda device number\n";
            exit(0);
        }
        deviceNumber = i;

        cudaSetDevice(deviceNumber);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceNumber);

        Catch::cout() <<"Device "<< deviceNumber <<": "<<deviceProp.name<<"\n";
    }
};

int main(int argc, char** argv)
{

    Catch::Session session;
    session.applyCommandLine(argc, argv, Catch::Session::OnUnusedOptions::Ignore);

    OtherOpt config;
    Catch::Clara::CommandLine<OtherOpt> cli;

    cli["-D"]["--device"]
        .describe( "Set cuda device" )
        .bind( &OtherOpt::setValidDeviceNumber, "deviceNumber");

    cli.parseInto( argc-1, argv+1, config );  //parse extra args (like cuda device)

    if(session.configData().showHelp) {
        Catch::cout() << "\ngpu_compression_lib specific options\n";
        cli.usage(Catch::cout(), session.configData().processName);
    }
    
    return session.run();
}
