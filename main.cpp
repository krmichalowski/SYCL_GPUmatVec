#include <iostream>
#include <hipSYCL/sycl.hpp>

namespace sycl = hipsycl::sycl;

int main()
{   
    int workGroups = 10;
    int workItemsPerGroup = 128;
    //initialize the host memory
    int size = 1280;
    double* x = new double[size];
    double* b = new double[size];
    int i = 0;
    for(i=0;i<size;i++)
    {
        x[i] = 1.0;
        b[i] = 0.0;
    }

    //selecting the gpu as device and creating a command queue for it
    sycl::gpu_selector gpu;
    sycl::queue queue(gpu);
    std::cout<<queue.get_device().get_info<sycl::info::device::name>()<<std::endl;

    //start the device scope, after which buffers will copy the results back to host memory
    {
        int tmpSize = workGroups * workItemsPerGroup * (size/workItemsPerGroup);
        double* tmpResults = new double[tmpSize];
        //initialize the buffers
        sycl::buffer<double, 1> xBuff(x, sycl::range<1>(size));
        sycl::buffer<double, 1> bBuff(b, sycl::range<1>(size));
        sycl::buffer<double, 1> tmpBuff(tmpResults, sycl::range<1>(tmpSize));

        //define the kernel to obtain tmp results
        auto matVecKernel = [&] (sycl::handler& cgh){
            //initialize global memory
            //part of x will later be copied to local memory of every work group, b will be stored in global memory
            auto x_access = xBuff.get_access<sycl::access::mode::read>(cgh);
            auto b_access = bBuff.get_access<sycl::access::mode::write>(cgh);
            auto tmp_access = tmpBuff.get_access<sycl::access::mode::discard_write>(cgh);
            //initialize local memory 
            //(using local memory for tmp results would also be benefitial but the size of local memory on my gpu is to small for that)
            //(will try to work around that by using one register variable to sum on and than write that to global mem, and repeat for every row)
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> xPart(sycl::range<1>(size/workGroups), cgh);
            int loadsPerWorkItem = size/(workGroups * workItemsPerGroup);
            int xPartSize = size/workGroups;
            cgh.parallel_for(sycl::nd_range<1>(workGroups * workItemsPerGroup, workGroups),
            [=] (sycl::nd_item<1> item){
                int groupId = item.get_group_linear_id();
                int localId = item.get_local_linear_id();
                //every work item loads part of x to local memory
                for(int i=0;i<loadsPerWorkItem;i++)
                {
                    xPart[localId * loadsPerWorkItem + i] = x_access[groupId * xPartSize + localId + i];
                }

                //fill part of tmpResults for this work item
            });
        };

        queue.submit(matVecKernel);
    }
}