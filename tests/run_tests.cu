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
        if( i < 0 || i > deviceCount )
            throw std::domain_error( "The device number is incorrect, please set valid cuda device number" );
        deviceNumber = i;

        cudaSetDevice(deviceNumber);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceNumber);

        std::cout<<"Device "<< deviceNumber <<": "<<deviceProp.name<<"\n";
    }
};

int main(int argc, char** argv)
{

    Catch::Session session;
    session.applyCommandLine(argc, argv, Catch::Session::OnUnusedOptions::Ignore);

    if (argc >= 2) {
        OtherOpt config;
        Catch::Clara::CommandLine<OtherOpt> cli;

        cli.bindProcessName( &OtherOpt::processName );

        cli["-d"]["--device"]
            .describe( "Set cuda device" )
            .bind( &OtherOpt::setValidDeviceNumber, "deviceNumber");

        cli["-?"]["-h"]["--help"]
            .describe( "display usage information" )
            .bind( &OtherOpt::showHelp );

        cli.parseInto( argc-1, argv+1, config );  //parse extra args (like cuda device)

        if(config.showHelp) {
            cli.usage(Catch::cout(), config.processName);
            return 0;
        }
    }
    
    return session.run();
}
