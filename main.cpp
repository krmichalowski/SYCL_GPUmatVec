#include <iostream>
#include <hipSYCL/sycl.hpp>

namespace sycl = hipsycl::sycl;

int main()
{   
    int workGroups = 10;
    int workItemsPerGroup = 128;
    //initialize the host memory
    int size = 57600;
    double* x = new double[size];
    double* b = new double[size];
    int i = 0;
    for(i=0;i<size;i++)
    {
        x[i] = 1.0;
    }

    //selecting the gpu as device and creating a command queue for it
    sycl::gpu_selector gpu;
    sycl::queue queue(gpu);
    std::cout<<queue.get_device().get_info<sycl::info::device::name>()<<std::endl;


    int tmpSize = workGroups * workItemsPerGroup * (size/workItemsPerGroup);
    double* tmpResults = new double[tmpSize];
    //start the device scope, after which buffers will copy the results back to host memory
    {
        //initialize the buffers
        sycl::buffer<double, 1> xBuff(x, sycl::range<1>(size));
        sycl::buffer<double, 1> bBuff(b, sycl::range<1>(size));
        sycl::buffer<double, 1> tmpBuff(tmpResults, sycl::range<1>(tmpSize));

        //define the kernel to obtain tmp results
        auto matVecKernel = [&] (sycl::handler& cgh){
            //initialize global memory
            //part of x will later be copied to local memory of every work group, b will be stored in global memory
            auto x_access = xBuff.get_access<sycl::access::mode::read>(cgh);
            auto tmp_access = tmpBuff.get_access<sycl::access::mode::discard_write>(cgh);
            //initialize local memory 
            //(using local memory for tmp results would also be benefitial but the size of local memory on my gpu is to small for that)
            //(will try to work around that by using one register variable to sum on and than write that to global mem, and repeat for every row)
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> xPart(sycl::range<1>(size/workGroups), cgh);
            int loadsPerWorkItem = size/(workGroups * workItemsPerGroup);
            int xPartSize = size/workGroups;
            int rows = size/workItemsPerGroup;
            int cols = size/workGroups;
            cgh.parallel_for(sycl::nd_range<1>(workGroups * workItemsPerGroup, workItemsPerGroup),
            [=] (sycl::nd_item<1> item){
                int groupId = item.get_group_linear_id();
                int localId = item.get_local_linear_id();
                //every work item loads part of x to local memorys
                int i;
                int j;
                for(i=0;i<loadsPerWorkItem;i++)
                {
                    xPart[localId * loadsPerWorkItem + i] = x_access[groupId * xPartSize + localId + i];
                }
                item.barrier(sycl::access::fence_space::local_space);
                //fill part of tmpResults for this work item
                int iRoot = localId * rows;
                int jRoot = groupId * cols;
                double accumulator = 0;
                double coeff;
                for(i=0;i<rows;i++)
                {
                    for(j=0;j<cols;j++)
                    {
                        coeff = (double)(iRoot + i + jRoot + j)/size;
                        accumulator = accumulator + coeff * xPart[j]; 
                    }
                    tmp_access[groupId * workItemsPerGroup * rows + localId * rows + i] = accumulator;
                    accumulator = 0;
                }
            });
        };
        
        auto sumTmpResultsKernel = [&] (sycl::handler& cgh){
            auto b_access = bBuff.get_access<sycl::access::mode::discard_write>(cgh);
            auto tmp_access = tmpBuff.get_access<sycl::access::mode::read>(cgh);

            int offset = size;
            int rowsPerItem = size/(workGroups * workItemsPerGroup);
            cgh.parallel_for(sycl::range<1>(workGroups * workItemsPerGroup),
            [=] (sycl::item<1> item){
                int globalId = item.get_linear_id();

                int i;
                int j;
                double accumulator = 0;
                for(i=0;i<rowsPerItem;i++)
                {
                    for(j=0;j<workGroups;j++)
                    {
                        accumulator = accumulator + tmp_access[globalId * rowsPerItem + i + j * offset];
                    }
                    b_access[globalId * rowsPerItem + i] = accumulator;
                    accumulator = 0;
                }
            });
        };

        queue.submit(matVecKernel);
        queue.submit(sumTmpResultsKernel);

    }
    /*
    for(i=0;i<size;i++)
    {
        std::cout<<b[i]<<"   "<<i<<std::endl;
    }
    */
    std::cout<<b[size/2]<<std::endl;
    std::cout<<b[size/5]<<std::endl;
    std::cout<<b[0]<<std::endl;
    delete[] x;
    delete[] b;
    delete[] tmpResults;
}