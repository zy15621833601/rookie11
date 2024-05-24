//***************************程序说明
//*************此程序已验证2*2 以正确
// linux 运行命令 mpicc -o cd mpiDD.c -lm  mpiexec -n 16 ./cd
// 修改时间：2024年4月23日
// 对流扩散（特征线法)
#include<math.h>
#include<stdio.h>
#include<malloc.h>
#include"mpi.h"
#define pi 4*atan(1)
#define delta 1e-10
// 右端项
double Rightside_f(double,double,double,double,double);
// 初始值
double Initial_c(double, double,double); 


// 快速算法
void Thomas_Algorithm(double *, double *, double *, double *, int );

// ***************************
void SloveCharacterx0(int, int ,double , double,double *,double *,double *,double *);
void SloveCharacterx(int, int ,double , double ,double *,double *,double *,double *,double *);
void SloveCharacterx1(int, int ,double , double ,double *,double *,double *,double *);

void SloveCharactery0(int, int ,double , double,double *,double *,double *,double *);
void SloveCharactery(int, int ,double , double ,double *,double *,double *,double *,double *);
void SloveCharactery1(int, int ,double , double ,double *,double *,double *,double *);
 
// **************************


// 求解 有区域分解C的算法

void SolveInteriorx0(int ,int , int *,double *,double *,int ,int ,double *,double *,
				    double ,double ,double , int ,double ,
					double *, double *);
void SolveInteriorx(int ,int , int *,double *,double *,int,int,double *,double *,
				    double ,double ,double , int ,double ,
					double *, double *);
void SolveInteriorx1(int ,int , int *,double *,double *,int,int ,double *,double *,
				    double ,double ,double , int ,double ,
					double *, double *);



// *****************求解内解***************

// ************y*********************
void SolveInteriory0(int,int , int *,double *,double *,int ,int ,double *,double *,
				    double ,double ,double , int ,double ,
					double *, double *,double *, double );
void SolveInteriory(int ,int , int *,double *,double *,int ,int ,double *,double *,
				    double ,double ,double , int ,double ,
					double *, double *,double *, double);
void SolveInteriory1(int ,int , int *,double *,double *,int ,int ,double *,double *,
				    double ,double ,double , int ,double ,
					double *, double *, double *, double );



// *****************求解内解***************

// 求解 无区域分解的解算法

char Name[30], Name1[30];
// 文件信息
 FILE *Oup, *Oup1; 

int main(int argc, char*argv[])
{
	int my_rank,my_size,tag;  // 进程的基本信息
	int i,j,n,M,MM; 
	int dest; // 进程编号
	int Nx,Ny;  // 空间的剖分点
	int Intex,km,*Akm,*AIkm;
	int ii,jj,kk,i1,i2,i3,j1,j2,j3;
	double t,T,L;
	double t1,t2,CPU,normC,dd0; // 时间函数的测试
	double Dx,Dy,Dt;
	int m1,m2;
    double Hx,Hy;
	double Ite1,norm11,Ite11;
	int normq1;
	double *x,*y,*xx,*yy;
	// 扩散和速度
	double *FC,*Vx,*Vy;
	double *Cexa,*C,*Cea;
	double *C1,*Vx1,*Vy1;
	double *C1old1,*C1old2,*C1mid,*C1mid1;
	double *ql,*qr,*Ql,*Qr;
	double *qd,*qo,*Qd,*Qo;
	double DD1,R,Dd;
	double *cl,*cr;
	double alpha; //质量误差
	
	MPI_Status status;


	MPI_Init(&argc,&argv);  // 初始化
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);// 进程号
	MPI_Comm_size(MPI_COMM_WORLD,&my_size);// 总进程数
	
	tag=0;
	
    alpha=0.5;
	
	dd0=1e-3;
	  
	// 空间剖分
	L=1;
	Nx=10;
	Ny=10;
	Dx=L/Nx;
	Dy=L/Ny;
	
	// 时间剖分
	
	t=1;

	T=1;
	MM=10000;
	Dt=T/MM;
	//Dt=pow(Dx,2)/4;

	M=(int)(ceil(t/Dt));
	

	
	// ******以下处理动态变量*********
	 x  = (double*)malloc((Nx+1)*sizeof(double));  // 存储x的坐标
     y  = (double*)malloc((Ny+1)*sizeof(double));  // 存储y的坐标
	 xx = (double*)malloc(Nx*sizeof(double));  // 存储x的坐标
     yy = (double*)malloc(Ny*sizeof(double));  // 存储y的坐标
     
	 // x y 方向的坐标
     for(i=0;i<Nx+1;i++) 
	 {
		 x[i]=i*Dx;

	 }
	 for(i=0;i<Ny+1;i++) 
	 {
		 y[i]=i*Dy;
	 }
	 for(i=0;i<Nx;i++) 
	 {
		 xx[i]=(i+0.5)*Dx;
	 }
	 for(i=0;i<Ny;i++) 
	 {
		 yy[i]=(i+0.5)*Dy;
	 }
	 

	m1=2;
	m2=2;
	Hx=m1*Dx;
	Hy=m2*Dy;

	km=2;
	
	 if ((Nx%km)==0)
	 {
		 Intex=(int)(ceil(Nx/km));
	 }
	 else
	 {
		 Intex=(int)(ceil(Nx/km))+1;
	 }
	 // 存放浓度内点信息
	 AIkm=(int *) malloc(km*sizeof(int));
	 for (i=0;i<km;i++)
	 {
		 if (i!=km-1)
		 {
			 AIkm[i]=Intex;
		 }
		 else
		 {
			 AIkm[i]=(Nx+1)-(km-1)*Intex;
		 }
	 }

	///////**************************************


	 
	 if (my_rank==0)
	 {
		 C      = (double*)malloc(((Nx+1)*(Ny+1))*sizeof(double)); // 浓度的全值
		 Cexa   = (double*)malloc(((Nx+1)*(Ny+1))*sizeof(double)); // 浓度的全值

		 
		 Vx     =(double*)malloc(((Nx+1)*(Ny+1))*sizeof(double));
		 Vy     =(double*)malloc(((Nx+1)*(Ny+1))*sizeof(double));
  
		

		
		
		 

		 // 浓度初始化
		 for(i=0;i<Ny+1;i++)
		 {   
			 for(j=0;j<Nx+1;j++)
			 {
				 C[i*(Nx+1)+j]   =Initial_c(x[j],y[i],0); 
				 Cexa[i*(Nx+1)+j]=Initial_c(x[j],y[i],t);
			 }
		 }
		 
		
		 
		 // 速度的处理
		 for(i=0;i<Ny+1;i++)
		 {
			 for(j=0;j<Nx+1;j++)
			 {

				 Vx[i*(Nx+1)+j]    =x[j]*(1-x[j]);
				 Vy[i*(Nx+1)+j]    =y[i]*(1-y[i]);
				
			 }
		  }

		  //
		
		  
		
// *************其他非0进程 浓度，速度和扩散 分配及传递            
		 for (i=1;i<my_size;i++)
		 {
			 dest=i;
			 j1=i/km; j2=AIkm[j1];
			 i1=i%km; i2=AIkm[i1];
             
			 MPI_Send(&i2,1,MPI_INT,dest,tag,MPI_COMM_WORLD);
			 MPI_Send(&j2,1,MPI_INT,dest,tag,MPI_COMM_WORLD);
			 
			

			 C1     = (double*)malloc((i2*j2)*sizeof(double)); // 子区域浓度值

		

			 Vx1    = (double*)malloc((i2*j2)*sizeof(double)); // 子区域x方向速度值
			 Vy1    = (double*)malloc((i2*j2)*sizeof(double)); // 子区域x方向速度值

			
			 // 处理编号顺序
			 i3=0;
			 for (kk=0;kk<i1;kk++)
			 {
				 i3=i3+AIkm[kk];
			 }
			 j3=0;
			 for (kk=0;kk<j1;kk++)
			 {
				 j3=j3+AIkm[kk];
			 }
			 
			 // 浓度的 处理分配
			 for (jj=0;jj<j2;jj++)
			 {
				 for (ii=0;ii<i2;ii++)
				 {
					 C1    [jj*i2+ii]  =C    [(jj+j3)*(Nx+1)+ii+i3];

					 Vx1   [jj*i2+ii]  =Vx   [(jj+j3)*(Nx+1)+ii+i3];
					 Vy1   [jj*i2+ii]  =Vy   [(jj+j3)*(Nx+1)+ii+i3];

				 }
			 }
			  
			 MPI_Send(C1,i2*j2,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);

			 MPI_Send(Vx1,i2*j2,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
			 MPI_Send(Vy1,i2*j2,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
	 
		
			 free(C1);free(Vx1);free(Vy1);
         }

 //*******************其他非0进程 浓度，速度，扩散分配。

//************************** 0 进程 浓度，速度，扩散分配**************
		 j1=0; j2=AIkm[j1];
		 i1=0; i2=AIkm[i1];
		 C1     = (double*)malloc((i2*j2)*sizeof(double)); // 子区域浓度值

		 Vx1    = (double*)malloc((i2*j2)*sizeof(double)); // 子区域x方向速度值
		 Vy1    = (double*)malloc((i2*j2)*sizeof(double)); // 子区域y方向速度值

		 C1old1  = (double*)malloc((i2*j2)*sizeof(double)); 

		 C1old2  = (double*)malloc((i2*j2)*sizeof(double)); 

		 C1mid  = (double*)malloc((i2*j2)*sizeof(double)); 
		 C1mid1  = (double*)malloc((i2*j2)*sizeof(double)); 

		

		 Cea    = (double*)malloc((i2*j2*M)*sizeof(double));
		 // 浓度的 处理分配
		 for (jj=0;jj<j2;jj++)
		 {
			 for (ii=0;ii<i2;ii++)
			 {
				 C1[jj*i2+ii]    =C    [jj*(Nx+1)+ii];
				 
				 Vx1[jj*i2+ii]   =Vx   [jj*(Nx+1)+ii];
				 Vy1[jj*i2+ii]   =Vy   [jj*(Nx+1)+ii];
			
			 }
		  }
		  		 
		 
		 
		 free(C);free(Vx);free(Vy);
	 }
	 else
	 {
		 
		 MPI_Recv(&i2,1,MPI_INT,0,tag,MPI_COMM_WORLD,&status);
		 MPI_Recv(&j2,1,MPI_INT,0,tag,MPI_COMM_WORLD,&status);

		 ii=i2*j2;

		 C1     = (double*)malloc(ii*sizeof(double));
		 Vx1    = (double*)malloc(ii*sizeof(double));
		 Vy1    = (double*)malloc(ii*sizeof(double));
		

		 Cea    = (double*)malloc((ii*M)*sizeof(double));

		 MPI_Recv(C1,ii,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);
		
		 MPI_Recv(Vx1,ii,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);
		 MPI_Recv(Vy1,ii,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);


		 C1old1  = (double*)malloc((i2*j2)*sizeof(double)); 

	     C1old2  = (double*)malloc((i2*j2)*sizeof(double)); 

		 C1mid  = (double*)malloc((i2*j2)*sizeof(double));   
		 C1mid1  = (double*)malloc((i2*j2)*sizeof(double));  

	 }

	
	 t1 = MPI_Wtime();



    //********************* 时间循环************
	for (n=0;n<M;n++)
	{
     
	 
        	
        for (jj=0;jj<j2;jj++)
		{
			for (ii=0;ii<i2;ii++)
			{
				C1mid [jj*i2+ii]=C1[jj*i2+ii];
				C1old1[jj*i2+ii]=C1[jj*i2+ii];
	            C1old2[jj*i2+ii]=C1[jj*i2+ii];
				Cea[n*(i2*j2)+jj*i2+ii]=C1[jj*i2+ii];
                C1mid1 [jj*i2+ii]=C1[jj*i2+ii];
                
			}
		} 
       
	

      normq1=5;Ite1=1.0;
     while (normq1<=1&&Ite1>10e-6)
	 {
        //***************  开始进行值的传递*************
		//**************  x 方向传播********************
			normq1=normq1+1;
		if (my_rank%km==0)
		{
		
            for (j=0;j<j2;j++)
			{
				C1mid [j*i2]= 0;	
			}
			
			// *************************
            
			cr=(double*)malloc(j2*sizeof(double));
			
			for (j=0;j<j2;j++)
			{
				cr[j]=C1mid1[j*i2+i2-1];				
			}
			

			MPI_Send(cr,j2,MPI_DOUBLE,my_rank+1,tag,MPI_COMM_WORLD);
			MPI_Recv(cr,j2,MPI_DOUBLE,my_rank+1,tag,MPI_COMM_WORLD,&status);

            SloveCharacterx0(i2, j2,Dt, Dx, C1,C1mid1,Vx1,cr); 


			// *************************

			ql=(double*)malloc(j2*sizeof(double));
			qr=(double*)malloc(j2*sizeof(double));
			

			for (j=0;j<j2;j++)
			{
				ql[j]=0;
				for (jj=0;jj<m1;jj++)
				{
					ql[j]=ql[j]+C1[j*i2+i2-1-jj];
				}
			}
			
			MPI_Recv(qr,j2,MPI_DOUBLE,my_rank+1,tag,MPI_COMM_WORLD,&status);

			cr=(double*)malloc(j2*sizeof(double));
			
			
			MPI_Recv(cr,j2,MPI_DOUBLE,my_rank+1,tag,MPI_COMM_WORLD,&status);



			Ql=(double*)malloc(j2*sizeof(double));
			Qr=(double*)malloc(j2*sizeof(double));
			for (j=0;j<j2;j++)
			{
				DD1=dd0*(1+(cr[j]+C1mid[j*i2+i2-1])/2);
				
			    Qr[j]=-DD1*(qr[j]-ql[j])/(m1*Hx);
				Ql[j]=0;
			}
			MPI_Send(Qr,j2,MPI_DOUBLE,my_rank+1,tag,MPI_COMM_WORLD);


			// ***********内点求解**************
			SolveInteriorx0(my_rank,km,AIkm,C1,C1mid,i2,j2, Ql,Qr,
				          Dt,Dx,Dy, n,dd0,x,y);

            

			free(Ql);free(Qr);free(ql);free(qr);
			free(cr);

		}
		else if (my_rank%km==km-1)
		{
			for (j=0;j<j2;j++)
			{
				C1mid[j*i2+i2-1]=0;
			}

			// *************************
            cl=(double*)malloc(j2*sizeof(double));
			
			
			for (j=0;j<j2;j++)
			{
				cl[j]=C1mid1[j*i2];				
			}

			MPI_Send(cl,j2,MPI_DOUBLE,my_rank-1,tag,MPI_COMM_WORLD);
			MPI_Recv(cl,j2,MPI_DOUBLE,my_rank-1,tag,MPI_COMM_WORLD,&status);

            SloveCharacterx1(i2, j2,Dt, Dx, C1,C1mid1,Vx1,cl); 

			// ***********************
	
			qr=(double*)malloc(j2*sizeof(double));
			
			for (j=0;j<j2;j++)
			{
				qr[j]=0;
                for (jj=0;jj<m1;jj++)
				{
					qr[j]=qr[j]+C1[j*i2+jj];
				}	
			}
			MPI_Send(qr,j2,MPI_DOUBLE,my_rank-1,tag,MPI_COMM_WORLD);

			for (j=0;j<j2;j++)
			{
				cl[j]=C1mid[j*i2];			
			}

			MPI_Send(cl,j2,MPI_DOUBLE,my_rank-1,tag,MPI_COMM_WORLD);

			Ql=(double*)malloc(j2*sizeof(double));
			Qr=(double*)malloc(j2*sizeof(double));
			MPI_Recv(Ql,j2,MPI_DOUBLE,my_rank-1,tag,MPI_COMM_WORLD,&status);

            for (j=0;j<j2;j++)
			{
				Qr[j]=0;

			}
            // ***********内点求解**************
			SolveInteriorx1(my_rank,km,AIkm,C1,C1mid,i2,j2, Ql,Qr,
				          Dt,Dx,Dy, n,dd0,x,y);

			free(Ql);free(Qr);free(qr);
			free(cl);

		}
		else 
		{
		

			
			// *************************
            cl=(double*)malloc(j2*sizeof(double));
			cr=(double*)malloc(j2*sizeof(double));
			
			for (j=0;j<j2;j++)
			{
				cl[j]=C1mid1[j*i2];
				cr[j]=C1mid1[j*i2+i2-1];				
			}

			MPI_Send(cl,j2,MPI_DOUBLE,my_rank-1,tag,MPI_COMM_WORLD);
			MPI_Recv(cl,j2,MPI_DOUBLE,my_rank-1,tag,MPI_COMM_WORLD,&status);
			MPI_Send(cr,j2,MPI_DOUBLE,my_rank+1,tag,MPI_COMM_WORLD);
			MPI_Recv(cr,j2,MPI_DOUBLE,my_rank+1,tag,MPI_COMM_WORLD,&status);

            SloveCharacterx(i2, j2,Dt, Dx, C1,C1mid1,Vx1,cl,cr); 

			// ********************

			ql=(double*)malloc(j2*sizeof(double));
			qr=(double*)malloc(j2*sizeof(double));
			
			for (j=0;j<j2;j++)
			{
				ql[j]=0;
				qr[j]=0;
				for (jj=0;jj<m1;jj++)
				{
					qr[j]=qr[j]+C1[j*i2+jj];
					ql[j]=ql[j]+C1[j*i2+i2-1-jj];
				}
				
			}
			MPI_Send(qr,j2,MPI_DOUBLE,my_rank-1,tag,MPI_COMM_WORLD);

			
			MPI_Recv(qr,j2,MPI_DOUBLE,my_rank+1,tag,MPI_COMM_WORLD,&status);

			Ql=(double*)malloc(j2*sizeof(double));
			Qr=(double*)malloc(j2*sizeof(double));

			cl=(double*)malloc(j2*sizeof(double));
			cr=(double*)malloc(j2*sizeof(double));
			
			for (j=0;j<j2;j++)
			{
				cl[j]=C1mid[j*i2];
				
			}

			MPI_Send(cl,j2,MPI_DOUBLE,my_rank-1,tag,MPI_COMM_WORLD);
			
			
			MPI_Recv(cr,j2,MPI_DOUBLE,my_rank+1,tag,MPI_COMM_WORLD,&status);

			for (j=0;j<j2;j++)
			{
				
				DD1=dd0*(1+(cr[j]+C1mid[j*i2+i2-1])/2);
			    Qr[j]=-DD1*(qr[j]-ql[j])/(m1*Hx);
			}
			MPI_Send(Qr,j2,MPI_DOUBLE,my_rank+1,tag,MPI_COMM_WORLD);
			MPI_Recv(Ql,j2,MPI_DOUBLE,my_rank-1,tag,MPI_COMM_WORLD,&status);

			// ***********内点求解**************
			SolveInteriorx(my_rank,km,AIkm,C1,C1mid,i2,j2, Ql,Qr,
				          Dt,Dx,Dy,n,dd0,x,y);

			free(Qr);free(Ql);
			free(ql);free(qr);
			free(cl);free(cr);
		}
		//******************** x方向传播结束**********************
		
		// **********边界化处理

         if (my_rank/km==0)
		{
			// ***边界化处理			
			for (i=0;i<i2;i++)
			{
				C1mid [i]= 0;	
			}
		 }
         if (my_rank/km==km-1)
		{	
			
			for (i=0;i<i2;i++)
			{
				C1mid [(j2-1)*i2+i]= 0;
			}
			
		 }


		 for (jj=0;jj<j2;jj++)
		 {
			 for (ii=0;ii<i2;ii++)
			 {
				 C1old1[jj*i2+ii]  =C1old2[jj*i2+ii];
	             C1old2[jj*i2+ii]  =C1mid [jj*i2+ii];
			 }
		  }
		 norm11=0;
		 
		 for (jj=0;jj<j2;jj++)
		 {
			 for (ii=0;ii<i2;ii++)
			 {
				 norm11=norm11+pow(C1old2[jj*i2+ii]-C1old1[jj*i2+ii],2)*Dx*Dy;
			 }
		  }
		 norm11=sqrt(norm11);

		 if (my_rank==0)
		 {
			Ite1=norm11;
			for (i=1;i<my_size;i++)
		    {
				dest=i;
				MPI_Recv(&norm11,1,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD,&status);
				Ite1=Ite1+norm11;
			}
			for (i=1;i<my_size;i++)
		    {
				dest=i;
				MPI_Send(&Ite1,1,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
			}
            
		 }
		 else
		 {
			 MPI_Send(&norm11,1,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);   
			 MPI_Recv(&Ite1,1,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);
		 }
		 
         Ite11=Ite1;
		
       }

	 	   
		
    
		for (jj=0;jj<j2;jj++)
		{
			for (ii=0;ii<i2;ii++)
			{
				C1old1[jj*i2+ii]=C1mid[jj*i2+ii];
	            C1old2[jj*i2+ii]=C1mid[jj*i2+ii];
				C1[jj*i2+ii]    =C1mid[jj*i2+ii];
				C1mid1[jj*i2+ii]    =C1mid[jj*i2+ii];
			}
		 }



		normq1=1;
        
		Ite11=1.0;

	   while (normq1<=1&&Ite11>1e-6)
	   {
		   normq1=normq1+1;
		//******************** y方向传播开始*********************
		if (my_rank/km==0)
		{
			// ***边界化处理			
			for (i=0;i<i2;i++)
			{
				C1mid [i]= 0;
				
			}
			// **********边界化处理



			// *************************
            
			cr=(double*)malloc(i2*sizeof(double));
			
			for (i=0;i<i2;i++)
			{
				cr[i]=C1mid1[(j2-1)*i2+i];				
			}

			MPI_Send(cr,i2,MPI_DOUBLE,my_rank+km,tag,MPI_COMM_WORLD);
			MPI_Recv(cr,i2,MPI_DOUBLE,my_rank+km,tag,MPI_COMM_WORLD,&status);

            SloveCharactery0(i2,j2,Dt, Dy, C1,C1mid1,Vy1,cr); 



			// ***************************
			qo=(double*)malloc(i2*sizeof(double));
			qd=(double*)malloc(i2*sizeof(double));
		
			for (i=0;i<i2;i++)
			{
				qd[i]=0;
				for (ii=0;ii<m2;ii++)
				{
					qd[i]=qd[i]+C1[(j2-1-ii)*i2+i];
				}
			}
			MPI_Recv(qo,i2,MPI_DOUBLE,my_rank+km,tag,MPI_COMM_WORLD,&status);


            
			
			MPI_Recv(cr,i2,MPI_DOUBLE,my_rank+km,tag,MPI_COMM_WORLD,&status);
		   

			Qd=(double*)malloc(i2*sizeof(double));
			Qo=(double*)malloc(i2*sizeof(double));
			for (i=0;i<i2;i++)
			{
				

				DD1=dd0*(1+(cr[i]+C1mid[(j2-1)*i2+i])/2);

			    Qo[i]=-DD1*(qo[i]-qd[i])/(m2*Hy);
				Qd[i]=0;
			}

			MPI_Send(Qo,i2,MPI_DOUBLE,my_rank+km,tag,MPI_COMM_WORLD);



			// ***********内点求解**************
			SolveInteriory0(my_rank,km,AIkm,C1,C1mid,i2,j2, Qd,Qo,
				          Dt,Dx,Dy,n,dd0,x,y,Cea,alpha);




			free(Qd);free(Qo);free(qo);free(qd);
			free(cr);


		}
		else if (my_rank/km==km-1)
		{	

 
      		// ***边界化处理			
			for (i=0;i<i2;i++)
			{
				C1mid [(j2-1)*i2+i]= 0;
				
			}
			// **********边界化处理

			// *************************
            cl=(double*)malloc(i2*sizeof(double));
			
			for (i=0;i<i2;i++)
			{
				cl[i]=C1mid1[i];				
			}

			MPI_Send(cl,i2,MPI_DOUBLE,my_rank-km,tag,MPI_COMM_WORLD);
			MPI_Recv(cl,i2,MPI_DOUBLE,my_rank-km,tag,MPI_COMM_WORLD,&status);

            SloveCharactery1(i2, j2,Dt, Dy, C1,C1mid1,Vy1,cl); 


			// ***********************
			qo=(double*)malloc(i2*sizeof(double));
			
			for (i=0;i<i2;i++)
			{
                 
				qo[i]=0;
				for (ii=0;ii<m2;ii++)
				{
					qo[i]=qo[i]+C1[ii*i2+i];
				}
			}
			MPI_Send(qo,i2,MPI_DOUBLE,my_rank-km,tag,MPI_COMM_WORLD);

			for (i=0;i<i2;i++)
			{
				cl[i]=C1mid[i];				
			}

			MPI_Send(cl,i2,MPI_DOUBLE,my_rank-km,tag,MPI_COMM_WORLD);

			Qo=(double*)malloc(i2*sizeof(double));
			Qd=(double*)malloc(i2*sizeof(double));

			MPI_Recv(Qd,i2,MPI_DOUBLE,my_rank-km,tag,MPI_COMM_WORLD,&status);
			for (j=0;j<i2;j++)
			{
				Qo[j]=0;
			}


            // ***********内点求解**************
			SolveInteriory1(my_rank,km,AIkm,C1,C1mid,i2,j2, Qd,Qo,
				          Dt,Dx,Dy, n, dd0,x,y,Cea,alpha);




			free(Qo);free(Qd);free(qo);
			free(cl);

		}
		else 
		{

			// *************************
            cl=(double*)malloc(i2*sizeof(double));
			cr=(double*)malloc(i2*sizeof(double));
			
			for (i=0;i<i2;i++)
			{
				cl[i]=C1mid1[i];
				cr[i]=C1mid1[(j2-1)*i2+i];				
			}

			MPI_Send(cl,i2,MPI_DOUBLE,my_rank-km,tag,MPI_COMM_WORLD);		
			MPI_Recv(cl,i2,MPI_DOUBLE,my_rank-km,tag,MPI_COMM_WORLD,&status);
			MPI_Send(cr,i2,MPI_DOUBLE,my_rank+km,tag,MPI_COMM_WORLD);
			MPI_Recv(cr,i2,MPI_DOUBLE,my_rank+km,tag,MPI_COMM_WORLD,&status);

            SloveCharactery(i2, j2, Dt, Dy, C1,C1mid1,Vy1,cl,cr); 

            // ********************
			qd=(double*)malloc(i2*sizeof(double));
			qo=(double*)malloc(i2*sizeof(double));
			
			for (i=0;i<i2;i++)
			{
				qd[i]=0;
				qo[i]=0;
				for (ii=0;ii<m2;ii++)
				{
					qd[i]=qd[i]+C1[(j2-1-ii)*i2+i];
					qo[i]=qo[i]+C1[ii*i2+i];
				}
			}
			MPI_Send(qo,i2,MPI_DOUBLE,my_rank-km,tag,MPI_COMM_WORLD);

			MPI_Recv(qo,i2,MPI_DOUBLE,my_rank+km,tag,MPI_COMM_WORLD,&status);

			for (i=0;i<i2;i++)
			{
				cl[i]=C1mid[i];
								
			}

			MPI_Send(cl,i2,MPI_DOUBLE,my_rank-km,tag,MPI_COMM_WORLD);		
			
			MPI_Recv(cr,i2,MPI_DOUBLE,my_rank+km,tag,MPI_COMM_WORLD,&status);
 
			Qd=(double*)malloc(i2*sizeof(double));
			Qo=(double*)malloc(i2*sizeof(double));
			for (j=0;j<i2;j++)
			{
				

				DD1=dd0*(1+(cr[j]+C1mid[(j2-1)*i2+j])/2);
				
			    Qo[j]=-DD1*(qo[j]-qd[j])/(m2*Hy);
			}

			MPI_Send(Qo,i2,MPI_DOUBLE,my_rank+km,tag,MPI_COMM_WORLD);
			MPI_Recv(Qd,i2,MPI_DOUBLE,my_rank-km,tag,MPI_COMM_WORLD,&status);


			// ***********内点求解**************

			SolveInteriory(my_rank,km,AIkm,C1,C1mid,i2,j2, Qd,Qo,
				          Dt,Dx,Dy,n,dd0,x,y,Cea,alpha);

			free(qd);free(qo);
			free(Qd); free(Qo);
			free(cl);free(cr);
			

		}
		//***********************y方向结束**********************


		// **********边界化处理
		if ((my_rank%km)==0)
		{   
			for (j=0;j<j2;j++)
			{
				C1mid [j*i2]= 0;	
			}
		}
		if ((my_rank%km)==km-1)
		{
			for (j=0;j<j2;j++)
			{
				C1mid[j*i2+i2-1]=0;
			}
		}
      

		for (jj=0;jj<j2;jj++)
		{
			for (ii=0;ii<i2;ii++)
			{
				C1old1[jj*i2+ii]  =C1old2[jj*i2+ii];
	            C1old2[jj*i2+ii]  =C1mid [jj*i2+ii];
			}
		 }
		 norm11=0;
		 
		 for (jj=0;jj<j2;jj++)
		 {
			 for (ii=0;ii<i2;ii++)
			 {
				 norm11=norm11+pow(C1old2[jj*i2+ii]-C1old1[jj*i2+ii],2)*Dx*Dy;
			 }
		  }
		 norm11=sqrt(norm11);

		 if (my_rank==0)
		 {
			Ite1=norm11;
			for (i=1;i<my_size;i++)
		    {
				dest=i;
				MPI_Recv(&norm11,1,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD,&status);
				Ite1=Ite1+norm11;
			}
			for (i=1;i<my_size;i++)
		    {
				dest=i;
				MPI_Send(&Ite1,1,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
			}
            
		 }
		 else
		 {
			 MPI_Send(&norm11,1,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);   
			 MPI_Recv(&Ite1,1,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&status);
		 }
		 
         Ite11=Ite1;

	}

	
		for (jj=0;jj<j2;jj++)
		{
			for (ii=0;ii<i2;ii++)
			{
				C1[jj*i2+ii] =C1mid[jj*i2+ii];

			}
		}

	}


	//*************************时间结束**********************
	t2 = MPI_Wtime();
// 估算并行时间
//	printf("%lf\n",t2-t1);
//     以下程序可以验证正确

	if (my_rank==0)
	{
		

		C=(double*)malloc((Nx+1)*(Ny+1)*sizeof(double));
		j1=0; j2=AIkm[j1];
		i1=0; i2=AIkm[i1];
		// 浓度的 处理分配
		for (jj=0;jj<j2;jj++)
		{
			for (ii=0;ii<i2;ii++)
			{
				 C[jj*(Nx+1)+ii]=C1[jj*i2+ii];
			}
		 }
		free(C1);
		for (i=1;i<my_size;i++)
		{
			dest=i;
			j1=i/km; j2=AIkm[j1];
			i1=i%km; i2=AIkm[i1];
			ii=i2*j2;
			C1 = (double*)malloc((ii)*sizeof(double)); // 子区域浓度值
			MPI_Recv(C1,ii,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD,&status);
			// 处理编号顺序
			i3=0;
			for (kk=0;kk<i1;kk++)
			{
				i3=i3+AIkm[kk];
			 }
			 j3=0;
			 for (kk=0;kk<j1;kk++)
			 {
				 j3=j3+AIkm[kk];
			 }
			 
			 // 浓度的 处理分配
			 for (jj=0;jj<j2;jj++)
			 {
				 for (ii=0;ii<i2;ii++)
				 {
					 C[(jj+j3)*(Nx+1)+ii+i3]=C1[jj*i2+ii];
					 
				 }
			 }
			 free(C1);
			 
		}
		normC=0;
		for (i=0;i<Ny+1;i++)
		{
			for (j=0;j<Nx+1;j++)
			{
				normC=normC+pow((C[i*(Nx+1)+j]-Cexa[i*(Nx+1)+j]),2)*Dx*Dy;
				//	printf("%d %lf %lf %lf\n",i*(Nx+1)+j,C[i*(Nx+1)+j],Cexa[i*(Nx+1)+j],
					//       C[i*(Nx+1)+j]-Cexa[i*(Nx+1)+j]);
			}
		}
		normC=sqrt(normC);
printf("%g\n",normC);	



		free(C); free(Cexa);
		

	}
	else
	{
		j1=my_rank/km; j2=AIkm[j1];
		i1=my_rank%km; i2=AIkm[i1];
		ii=i2*j2;
		MPI_Send(C1,ii,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);

		
		free(C1);

	}

	free(x);free(y);free(AIkm);
	free(xx);free(yy);
	MPI_Finalize();


	
}


// 子函数的调用

// 函数调用

double Initial_c(double x, double y,double t) 
{
   double c0;
   
   c0=pow(t,2)*pow(x,2)*pow(1-x,2)*pow(y,2)*pow(1-y,2);

   
   return c0;
}


// 浓度右端项
double Rightside_f(double alpha,double dd0, double x, double y, double t)
{
   double ff; 
   double D1,D2,DC,B1,B2;
  
   D1=dd0;
   D2=dd0;
   DC=1;
   B1=1;
   B2=1;

  
    
    ff=2*pow(x,2)*pow(1-x,2)*pow(y,2)*pow(1-y,2)*(t+DC*pow(t,2-alpha)/tgamma(3-alpha)+B1*pow(t,2)*(2-2*x-2*y))
		-4*D1*pow(t,4)*pow(x,2)*pow(1-x,2)*pow(y,2)*pow(1-y,2)*
(pow(y,2)*pow(1-y,2)*pow(1-2*x,2)+pow(x,2)*pow(1-x,2)*pow(1-2*y,2))
		-2*D1*pow(t,2)*(1+pow(t,2)*pow(x,2)*pow(1-x,2)*pow(y,2)*pow(1-y,2))*
(pow(y,2)*pow(1-y,2)*(1-6*x+6*pow(x,2))+pow(x,2)*pow(1-x,2)*(1-6*y+6*pow(y,2)));

 
   return ff;
}


void Thomas_Algorithm(double *L, double *D, double *U, double *d, int nn)
{
	int i; double temp,temp1,temp2;
    double *P,*Q;
        
    P=(double*)malloc((nn+1)*sizeof(double));
    Q=(double*)malloc((nn+1)*sizeof(double));
        
    P[0]=0.0;
    Q[0]=0.0;
        
    for (i=0;i<nn;i++)
    {
		temp1  = L[i]*P[i];
		temp   = D[i]+temp1;
        P[i+1] = -U[i]/temp;
        temp2  = L[i]*Q[i];
        Q[i+1] = (d[i]-temp2)/temp;          
     }

     /*Backward procedure*/    
     for(i=nn-1;i>=0;i--) 
     {       
		 temp=P[i+1]*d[i+1];
         d[i]=temp+Q[i+1];
      /*printf("%lf\n",d[i]);*/
	 }
	 free(P);free(Q);
}


// **********x***************************
 void SloveCharacterx0(int i2, int j2,double Dt, double Dx,double *C1,double *C1mid1,double *Vx1,double *cr)
 {
	int ii,jj;
	double *tempC;
	double aa,bb,cc,vv;

	tempC=(double*)malloc(i2*sizeof(double));

	for (jj=0;jj<j2;jj++)
	{
		for (ii=0;ii<i2;ii++)
		{
			tempC[ii]=C1mid1[jj*i2+ii];
		}
		for (ii=0;ii<i2;ii++)
		{
			vv=-Vx1[jj*i2+ii]*Dt/Dx;
			if (ii==0)
			{
				C1[jj*i2+ii]=tempC[ii];
			}
			else if (ii==i2-1)
			{
				aa=tempC[ii-1];
				bb=tempC[ii];
				cc=cr[jj];
				C1[jj*i2+ii]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
			else
			{
				aa=tempC[ii-1];
				bb=tempC[ii];
				cc=tempC[ii+1];
				C1[jj*i2+ii]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
		}
	}
	free(tempC);
 }

void SloveCharacterx(int i2, int j2,double Dt, double Dx,double *C1,double *C1mid1,double *Vx1,double *cl,double *cr)
 {
	int ii,jj;
	double *tempC;
	double aa,bb,cc,vv;

	tempC=(double*)malloc(i2*sizeof(double));

	for (jj=0;jj<j2;jj++)
	{
		for (ii=0;ii<i2;ii++)
		{
			tempC[ii]=C1mid1[jj*i2+ii];
		}
		for (ii=0;ii<i2;ii++)
		{
			vv=-Vx1[jj*i2+ii]*Dt/Dx;
			if (ii==0)
			{
				aa=cl[jj];
				bb=tempC[ii];
				cc=tempC[ii+1];
				C1[jj*i2+ii]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
			else if (ii==i2-1)
			{
				aa=tempC[ii-1];
				bb=tempC[ii];
				cc=cr[jj];
				C1[jj*i2+ii]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
			else
			{
				aa=tempC[ii-1];
				bb=tempC[ii];
				cc=tempC[ii+1];
				C1[jj*i2+ii]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
		}
	}
	free(tempC);
 }

void SloveCharacterx1(int i2, int j2,double Dt, double Dx,double *C1,double *C1mid1,double *Vx1,double *cl)
 {
	int ii,jj;
	double *tempC;
	double aa,bb,cc,vv;

	tempC=(double*)malloc(i2*sizeof(double));

	for (jj=0;jj<j2;jj++)
	{
		for (ii=0;ii<i2;ii++)
		{
			tempC[ii]=C1mid1[jj*i2+ii];
		}
		for (ii=0;ii<i2;ii++)
		{
			vv=-Vx1[jj*i2+ii]*Dt/Dx;
			if (ii==0)
			{
				aa=cl[jj];
				bb=tempC[ii];
				cc=tempC[ii+1];
				C1[jj*i2+ii]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
			else if (ii==i2-1)
			{
				C1[jj*i2+ii]=tempC[ii];
			}
			else
			{
				aa=tempC[ii-1];
				bb=tempC[ii];
				cc=tempC[ii+1];
				C1[jj*i2+ii]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
		}
	}
	free(tempC);
 }
// *************x******************

// **********y***************************
 void SloveCharactery0(int i2, int j2,double Dt, double Dy,double *C1,double *C1mid1,double *Vy1,double *cr)
 {
	int ii,jj;
	double *tempC;
	double aa,bb,cc,vv;

	tempC=(double*)malloc(j2*sizeof(double));

	for (jj=0;jj<i2;jj++)
	{
		for (ii=0;ii<j2;ii++)
		{
			tempC[ii]=C1mid1[ii*i2+jj];
		}
		for (ii=0;ii<j2;ii++)
		{
			vv=-Vy1[ii*i2+jj]*Dt/Dy;
			if (ii==0)
			{
				C1[ii*i2+jj]=tempC[ii];
			}
			else if (ii==j2-1)
			{
				aa=tempC[ii-1];
				bb=tempC[ii];
				cc=cr[jj];
				C1[ii*i2+jj]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
			else
			{
				aa=tempC[ii-1];
				bb=tempC[ii];
				cc=tempC[ii+1];
				C1[ii*i2+jj]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
		}
	}
	free(tempC);
 }

void SloveCharactery(int i2, int j2,double Dt, double Dy,double *C1,double *C1mid1,double *Vy1,double *cl,double *cr)
 {
	int ii,jj;
	double *tempC;
	double aa,bb,cc,vv;

	tempC=(double*)malloc(j2*sizeof(double));

	for (jj=0;jj<i2;jj++)
	{
		for (ii=0;ii<j2;ii++)
		{
			tempC[ii]=C1mid1[ii*i2+jj];
		}
		for (ii=0;ii<j2;ii++)
		{
			vv=-Vy1[ii*i2+jj]*Dt/Dy;
			if (ii==0)
			{
				aa=cl[jj];
				bb=tempC[ii];
				cc=tempC[ii+1];
				C1[ii*i2+jj]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
			else if (ii==j2-1)
			{
				aa=tempC[ii-1];
				bb=tempC[ii];
				cc=cr[jj];
				C1[ii*i2+jj]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
			else
			{
				aa=tempC[ii-1];
				bb=tempC[ii];
				cc=tempC[ii+1];
				C1[ii*i2+jj]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
		}
	}
	free(tempC);
 }

void SloveCharactery1(int i2, int j2,double Dt, double Dy,double *C1,double *C1mid1,double *Vy1,double *cl)
 {
	int ii,jj;
	double *tempC;
	double aa,bb,cc,vv;

	tempC=(double*)malloc(j2*sizeof(double));

	for (jj=0;jj<i2;jj++)
	{
		for (ii=0;ii<j2;ii++)
		{
			tempC[ii]=C1mid1[ii*i2+jj];
		}
		for (ii=0;ii<j2;ii++)
		{
			vv=-Vy1[ii*i2+jj]*Dt/Dy;
			if (ii==0)
			{
				aa=cl[jj];
				bb=tempC[ii];
				cc=tempC[ii+1];
				C1[ii*i2+jj]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
			else if (ii==j2-1)
			{
				C1[ii*i2+jj]=tempC[ii];
			}
			else
			{
				aa=tempC[ii-1];
				bb=tempC[ii];
				cc=tempC[ii+1];
				C1[ii*i2+jj]=pow(vv,2)*(aa+cc)/2+(1-pow(vv,2))*bb+(cc-aa)*vv/2;
			}
		}
	}
	free(tempC);
 }
// ***********************y*************************



// **********计算内点************************************

void SolveInteriorx0(int my_rank,int km, int *AIkm,double *C1,double *C1mid,int i2,int j2,double *Ql,double *Qr,
				    double Dt,double Dx,double Dy, int n,double dd0,
					double *x, double *y)
// *****************求解内解***************
{
	int I1, I2;
	int ii,jj,j;
	double *L,*D,*U,*d,*Cc,*Ccmid;
	double DD,DD0,DD1,R,R1;
	double aaa,aab,aac;
	I1=0;I2=0;
	ii=my_rank%km;
	jj=my_rank/km;
	for (j=0;j<ii;j++)
	{
		I1=I1+AIkm[j];
	}
	for (j=0;j<jj;j++)
	{
		I2=I2+AIkm[j];
	}
	L=(double*)malloc(i2*sizeof(double));
	D=(double*)malloc(i2*sizeof(double));
	U=(double*)malloc(i2*sizeof(double));
	d=(double*)malloc(i2*sizeof(double));
	Cc=(double*)malloc(i2*sizeof(double));
	Ccmid=(double*)malloc(i2*sizeof(double));

	
	
	for (j=0;j<j2;j++)
	{
		for (ii=0;ii<i2;ii++)
		{
			L[ii]=0.0;
			D[ii]=0.0;
			U[ii]=0.0;
			d[ii]=0.0;       
			Cc[ii]=C1[j*i2+ii];
			Ccmid[ii]=C1mid[j*i2+ii];

		}
		for (ii=0;ii<i2-1;ii++)
		{
			if (ii==0)
			{
			
                aaa=Ccmid[ii];
				aab=Ccmid[ii+1];
				aac=Ccmid[ii+2];

                DD0 = dd0*(1+(aaa+aab)/2);
				DD1 = dd0*(1+(aab+aac)/2);

				R =DD0*Dt/(pow(Dx,2));
				R1=DD1*Dt/(pow(Dx,2));
				
				D[ii]=1+R+R1;
				U[ii]=-R1;
			}
			else if(ii==i2-2)
			{
				
		        aaa=Ccmid[ii];
				aab=Ccmid[ii+1];
				
				
				DD = dd0*(1+(aaa+aab)/2);

				R=DD*Dt/(pow(Dx,2));
				
				L[ii]=-R;
				D[ii]=1+R;

			}
			else
			{
				
                aaa=Ccmid[ii];
				aab=Ccmid[ii+1];
				aac=Ccmid[ii+2];

				DD0 = dd0*(1+(aaa+aab)/2);
				DD1 = dd0*(1+(aab+aac)/2);


				R =DD0*Dt/(pow(Dx,2));
				R1=DD1*Dt/(pow(Dx,2));
				
				L[ii]=-R;
				D[ii]=1+R+R1;
				U[ii]=-R1;
			}
			d[ii]=Cc[ii+1];

		}
		
				
		d[0]=d[0]+Dt/Dx*Ql[j];
		d[i2-2]=d[i2-2]-Dt/Dx*Qr[j];

		Thomas_Algorithm(L,D,U,d,i2-1);
              
		for(ii=0;ii<i2-1;ii++)  
		{		
			C1mid[j*i2+ii+1]=d[ii];
		}
	}

	
	free(L);free(D);free(U);
	free(d);free(Cc);

	
}


void SolveInteriorx(int my_rank,int km, int *AIkm,double *C1,double *C1mid,int i2,int j2,double *Ql,double *Qr,
				    double Dt,double Dx,double Dy, int n,double dd0,
					double *x, double *y)
// *****************求解内解***************
{
	int I1, I2;
	int ii,jj,j;
	double *L,*D,*U,*d,*Cc,*Ccmid;
	double DD,DD0,DD1,R,R1;
	double aaa,aab,aac;

	I1=0;I2=0;
	ii=my_rank%km;
	jj=my_rank/km;
	for (j=0;j<ii;j++)
	{
		I1=I1+AIkm[j];
	}
	for (j=0;j<jj;j++)
	{
		I2=I2+AIkm[j];
	}
	L=(double*)malloc(i2*sizeof(double));
	D=(double*)malloc(i2*sizeof(double));
	U=(double*)malloc(i2*sizeof(double));
	d=(double*)malloc(i2*sizeof(double));
	Cc=(double*)malloc(i2*sizeof(double));
	Ccmid=(double*)malloc(i2*sizeof(double));
	

	
	for (j=0;j<j2;j++)
	{
		for (ii=0;ii<i2;ii++)
		{
			L[ii]=0.0;
			D[ii]=0.0;
			U[ii]=0.0;
			d[ii]=0.0;       
			Cc[ii]=C1[j*i2+ii];
			Ccmid[ii]=C1mid[j*i2+ii];

			if (ii==0)
			{
				
                aaa=Ccmid[ii];
				aab=Ccmid[ii+1];
				

                DD = dd0*(1+(aaa+aab)/2);
				


				R=DD*Dt/(pow(Dx,2));
				
				D[ii]=1+R;
				U[ii]=-R;
			}
			else if(ii==i2-1)
			{
				aaa=Ccmid[ii-1];
				aab=Ccmid[ii];

				DD =dd0*(1+(aaa+aab)/2);

				R=DD*Dt/(pow(Dx,2));
				
				L[ii]=-R;
				D[ii]=1+R;

			}
			else
			{
				aaa=Ccmid[ii-1];
				aab=Ccmid[ii];
				aac=Ccmid[ii+1];
				
				DD0 = dd0*(1+(aaa+aab)/2);
				DD1 = dd0*(1+(aac+aab)/2);

				R =DD0*Dt/(pow(Dx,2));
				R1=DD1*Dt/(pow(Dx,2));
				
				L[ii]=-R;
				D[ii]=1+R+R1;
				U[ii]=-R1;
			}
			d[ii]=Cc[ii];

		}
		d[0]=d[0]+Dt/Dx*Ql[j];
		d[i2-1]=d[i2-1]-Dt/Dx*Qr[j];

		Thomas_Algorithm(L,D,U,d,i2);
              
		for(ii=0;ii<i2;ii++)  
		{
			C1mid[j*i2+ii]=d[ii];
		}
	}

	
	free(L);free(D);free(U);
	free(d);free(Cc);

	
}


void SolveInteriorx1(int my_rank,int km, int *AIkm,double *C1,double *C1mid,int i2,int j2,double *Ql,double *Qr,
				    double Dt,double Dx,double Dy, int n,double dd0,
					double *x, double *y)
// *****************求解内解***************
{
	int I1, I2;
	int ii,jj,j;
	double *L,*D,*U,*d,*Cc,*Ccmid;
	double DD,DD0,DD1,R,R1;
	double aaa,aab,aac;
	I1=0;I2=0;
	ii=my_rank%km;
	jj=my_rank/km;
	for (j=0;j<ii;j++)
	{
		I1=I1+AIkm[j];
	}
	for (j=0;j<jj;j++)
	{
		I2=I2+AIkm[j];
	}
	L=(double*)malloc(i2*sizeof(double));
	D=(double*)malloc(i2*sizeof(double));
	U=(double*)malloc(i2*sizeof(double));
	d=(double*)malloc(i2*sizeof(double));
	Cc=(double*)malloc(i2*sizeof(double));
	Ccmid=(double*)malloc(i2*sizeof(double));

	
	for (j=0;j<j2;j++)
	{
		for (ii=0;ii<i2;ii++)
		{
			L[ii]=0.0;
			D[ii]=0.0;
			U[ii]=0.0;
			d[ii]=0.0;       
			Cc[ii]=C1[j*i2+ii];
			Ccmid[ii]=C1mid[j*i2+ii];
		}
		for (ii=0;ii<i2-1;ii++)
		{
			if (ii==0)
			{
				
				aaa=Ccmid[ii];
				aab=Ccmid[ii+1];
				
                DD = dd0*(1+(aaa+aab)/2);

				R=DD*Dt/(pow(Dx,2));
				
				D[ii]=1+R;
				U[ii]=-R;
			}
			else if(ii==i2-2)
			{
				
				aaa=Ccmid[ii-1];
				aab=Ccmid[ii];
				aac=Ccmid[ii+1];
				
				DD0 = dd0*(1+(aaa+aab)/2);
				DD1 = dd0*(1+(aac+aab)/2);


				R =DD0*Dt/(pow(Dx,2));
				R1=DD1*Dt/(pow(Dx,2));
				
				L[ii]=-R;
				D[ii]=1+R+R1;

			}
			else
			{
				
				aaa=Ccmid[ii-1];
				aab=Ccmid[ii];
				aac=Ccmid[ii+1];
				
				DD0 = dd0*(1+(aaa+aab)/2);
				DD1 = dd0*(1+(aac+aab)/2);


				R =DD0*Dt/(pow(Dx,2));
				R1=DD1*Dt/(pow(Dx,2));
				
				L[ii]=-R;
				D[ii]=1+R+R1;
				U[ii]=-R1;
			}
			d[ii]=Cc[ii];

		}
		d[0]=d[0]+Dt/Dx*Ql[j];


		d[i2-2]=d[i2-2]-Dt/Dx*Qr[j];

		Thomas_Algorithm(L,D,U,d,i2-1);
              
		for(ii=0;ii<i2-1;ii++)  
		{ 
			C1mid[j*i2+ii]=d[ii];
		}
	}

	
	free(L);free(D);free(U);
	free(d);free(Cc);	
}

//  ******************x************************

// ****************y方向内点************
// ************y*********************
void SolveInteriory0(int my_rank,int km, int *AIkm,double *C1,double *C1mid,int i2,int j2,double *Qd,double *Qo,
				    double Dt,double Dx,double Dy, int n,double dd0,
					double *x, double *y,double *Cea, double alpha)

// *****************求解内解***************
{
	int ii,jj,j;
	double *L,*D,*U,*d,*Cc,*ggab,*ggac,*tempcc,*FFc;
	double DD,DD0,DD1,R,R1,aabb;
	double aab,aac,aaa,*Ccmid;
	int I1,I2,nn,kkk;

    aabb=pow(Dt,1-alpha)/tgamma(2-alpha);
	I1=0;I2=0;
	ii=my_rank%km;
	jj=my_rank/km;
	for (j=0;j<ii;j++)
	{
		I1=I1+AIkm[j];
	}
	for (j=0;j<jj;j++)
	{
		I2=I2+AIkm[j];
	}

	ggab=(double*)malloc((n+1)*sizeof(double));
	ggac=(double*)malloc((n+1)*sizeof(double));
	tempcc=(double*)malloc((i2*j2)*sizeof(double));
	
	L=(double*)malloc(j2*sizeof(double));
	D=(double*)malloc(j2*sizeof(double));
	U=(double*)malloc(j2*sizeof(double));
	d=(double*)malloc(j2*sizeof(double));
	Cc=(double*)malloc(j2*sizeof(double));
	Ccmid=(double*)malloc(j2*sizeof(double));
	FFc=(double*)malloc(j2*sizeof(double));
	
	for (j=0;j<j2;j++)
	{
		for (ii=0;ii<i2;ii++)
		{
			tempcc[j*i2+ii]=0;
		}
	}

	for (kkk=0;kkk<n+1;kkk++)
	{
	  ggab[kkk]=pow(n+1-kkk,1-alpha)-pow(n-kkk,1-alpha);
	}

	for (nn=0;nn<n+1;nn++)
	{
		if (nn==0)
		{
			ggac[nn]=aabb*ggab[nn];
		}
		else 
		{
		  ggac[nn]=aabb*(ggab[nn]-ggab[nn-1]);
		}
		for (j=0;j<i2;j++)
		{
			for (ii=0;ii<j2;ii++)
			{
				tempcc[ii*i2+j]=tempcc[ii*i2+j]+ggac[nn]*Cea[nn*(i2*j2)+ii*i2+j];
			}
		}
	}


	for (j=0;j<i2;j++)
	{
		for (ii=0;ii<j2;ii++)
		{
			L[ii]=0.0;
			D[ii]=0.0;
			U[ii]=0.0;
			d[ii]=0.0;       
			Cc[ii]=C1[ii*i2+j];
			Ccmid[ii]=C1mid[ii*i2+j];
			FFc[ii]=tempcc[ii*i2+j];
		}
		for (ii=0;ii<j2-1;ii++)
		{
			if (ii==0)
			{
				
				aaa=Ccmid[ii];
				aab=Ccmid[ii+1];
				aac=Ccmid[ii+2];

				DD0=dd0*(1+(aaa+aab)/2);

				DD1=dd0*(1+(aac+aab)/2);
				
				R =DD0*Dt/(pow(Dy,2));
				R1=DD1*Dt/(pow(Dy,2));
				
				D[ii]=1+aabb+R+R1;

				U[ii]=-R1;
			}
			else if(ii==j2-2)
			{

				aaa=Ccmid[ii];
				aab=Ccmid[ii+1];
				

				DD=dd0*(1+(aaa+aab)/2);

				
				
				R  =DD*Dt/(pow(Dy,2));
				
				L[ii]=-R;
				D[ii]=1+aabb+R;

			}
			else
			{
				
                aaa=Ccmid[ii];
				aab=Ccmid[ii+1];
				aac=Ccmid[ii+2];

				DD0=dd0*(1+(aaa+aab)/2);

				DD1=dd0*(1+(aac+aab)/2);





				R =DD0*Dt/(pow(Dy,2));
				R1=DD1*Dt/(pow(Dy,2));
				
				L[ii]=-R;
				D[ii]=1+aabb+R+R1;

				U[ii]=-R1;
			}
			d[ii]=Cc[ii+1]+FFc[ii+1]+Dt*Rightside_f(alpha,dd0,x[I1+j],y[I2+ii+1],(n+1)*Dt);


		}
		
	
		d[0]=d[0]+Dt/Dy*Qd[j];
		d[j2-2]=d[j2-2]-Dt/Dy*Qo[j];

		Thomas_Algorithm(L,D,U,d,j2-1);
              
		for(ii=0;ii<j2-1;ii++)  
		{
			C1mid[(ii+1)*i2+j]=d[ii];
			//	printf("%d %d %d %d %lf\n",my_rank,j,ii, (ii+1)*i2+j,C1mid[(ii+1)*i2+j]);
		}
	}
	free(L);free(D);free(U);
	free(d);free(Cc);
	free(ggab);free(ggac);free(tempcc);free(FFc);

}


void SolveInteriory1(int my_rank,int km, int *AIkm,double *C1,double *C1mid,int i2,int j2,double *Qd,double *Qo,
				    double Dt,double Dx,double Dy, int n,double dd0,
					double *x, double *y, double *Cea, double alpha)
// *****************求解内解***************
{
	int ii,jj,j;
	double *L,*D,*U,*d,*Cc,*ggab,*ggac,*tempcc,*FFc;
	double DD,DD0,DD1,R,R1,aabb;
	int I1,I2;
	double aaa,aab,aac,*Ccmid;
	int kkk,nn;

	aabb=pow(Dt,1-alpha)/tgamma(2-alpha);

	I1=0;I2=0;
	ii=my_rank%km;
	jj=my_rank/km;
	for (j=0;j<ii;j++)
	{
		I1=I1+AIkm[j];
	}
	for (j=0;j<jj;j++)
	{
		I2=I2+AIkm[j];
	}


	L=(double*)malloc(j2*sizeof(double));
	D=(double*)malloc(j2*sizeof(double));
	U=(double*)malloc(j2*sizeof(double));
	d=(double*)malloc(j2*sizeof(double));
	Cc=(double*)malloc(j2*sizeof(double));
    Ccmid=(double*)malloc(j2*sizeof(double));
   
	ggab=(double*)malloc((n+1)*sizeof(double));
	ggac=(double*)malloc((n+1)*sizeof(double));
	tempcc=(double*)malloc((i2*j2)*sizeof(double));
	FFc=(double*)malloc(j2*sizeof(double));
	


	for (j=0;j<j2;j++)
	{
		for (ii=0;ii<i2;ii++)
		{
			tempcc[j*i2+ii]=0;
		}
	}


	for (kkk=0;kkk<n+1;kkk++)
	{
	  ggab[kkk]=pow(n+1-kkk,1-alpha)-pow(n-kkk,1-alpha);
	}

	for (nn=0;nn<n+1;nn++)
	{
		if (nn==0)
		{
			ggac[nn]=aabb*ggab[nn];
		}
		else 
		{
		  ggac[nn]=aabb*(ggab[nn]-ggab[nn-1]);
		}
		for (j=0;j<i2;j++)
		{
			for (ii=0;ii<j2;ii++)
			{
				tempcc[ii*i2+j]=tempcc[ii*i2+j]+ggac[nn]*Cea[nn*(i2*j2)+ii*i2+j];
			}
		}
	}


	for (j=0;j<i2;j++)
	{
		for (ii=0;ii<j2;ii++)
		{
			L[ii]=0.0;
			D[ii]=0.0;
			U[ii]=0.0;
			d[ii]=0.0;       
			Cc[ii]=C1[ii*i2+j];
			Ccmid[ii]=C1mid[ii*i2+j];
			FFc[ii]=tempcc[ii*i2+j];
		}
		for (ii=0;ii<j2-1;ii++)
		{
			if (ii==0)
			{
				
			    aaa=Ccmid[ii];
				aab=Ccmid[ii+1];
				

				DD=dd0*(1+(aaa+aab)/2);

			
				R=DD*Dt/(pow(Dy,2));
				
				D[ii]=1+aabb+R;
				U[ii]=-R;
			}
			else if(ii==j2-2)
			{
				aaa=Ccmid[ii-1];
				aab=Ccmid[ii];
				aac=Ccmid[ii+1];

				DD0=dd0*(1+(aaa+aab)/2);

				DD1 =dd0*(1+(aac+aab)/2);

				R =DD0*Dt/(pow(Dy,2));
				R1=DD1*Dt/(pow(Dy,2));
				
				L[ii]=-R;
				D[ii]=1+aabb+R+R1;

			}
			else
			{
				aaa=Ccmid[ii-1];
				aab=Ccmid[ii];
				aac=Ccmid[ii+1];

				DD0=dd0*(1+(aaa+aab)/2);

				DD1=dd0*(1+(aac+aab)/2);

				R =DD0*Dt/(pow(Dy,2));
				R1=DD1*Dt/(pow(Dy,2));
				
				L[ii]=-R;
				D[ii]=1+aabb+R+R1;
				U[ii]=-R1;
			}
			d[ii]=Cc[ii]+FFc[ii]+Dt*Rightside_f(alpha,dd0,x[I1+j],y[I2+ii],(n+1)*Dt);

		}
		d[0]=d[0]+Dt/Dy*Qd[j];

	
		d[j2-2]=d[j2-2]-Dt/Dy*Qo[j];

		Thomas_Algorithm(L,D,U,d,j2-1);
              
		for(ii=0;ii<j2-1;ii++)  
		{
			C1mid[ii*i2+j]=d[ii];

		}
	}
	free(L);free(D);free(U);
	free(d);free(Cc);
	free(ggab);free(ggac);free(tempcc);free(FFc);

}
void SolveInteriory(int my_rank,int km, int *AIkm,double *C1,double *C1mid,int i2,int j2,double *Qd,double *Qo,
				    double Dt,double Dx,double Dy, int n,double dd0,
					double *x, double *y,double *Cea, double alpha)
// *****************求解内解***************
{
	int ii,jj,j;
	double *L,*D,*U,*d,*Cc,*ggab,*ggac,*tempcc,*FFc;
	double DD,DD0,DD1,R,R1,aabb;
	int I1,I2;
	double aaa,aab,aac,*Ccmid;
    int kkk,nn;

    aabb=pow(Dt,1-alpha)/tgamma(2-alpha);

	I1=0;I2=0;
	ii=my_rank%km;
	jj=my_rank/km;
	for (j=0;j<ii;j++)
	{
		I1=I1+AIkm[j];
	}
	for (j=0;j<jj;j++)
	{
		I2=I2+AIkm[j];
	}


	L=(double*)malloc(j2*sizeof(double));
	D=(double*)malloc(j2*sizeof(double));
	U=(double*)malloc(j2*sizeof(double));
	d=(double*)malloc(j2*sizeof(double));
	Cc=(double*)malloc(j2*sizeof(double));
	Ccmid=(double*)malloc(j2*sizeof(double));

	ggab=(double*)malloc((n+1)*sizeof(double));
	ggac=(double*)malloc((n+1)*sizeof(double));
	tempcc=(double*)malloc((i2*j2)*sizeof(double));
	FFc=(double*)malloc(j2*sizeof(double));

	for (j=0;j<j2;j++)
	{
		for (ii=0;ii<i2;ii++)
		{
			tempcc[j*i2+ii]=0;
		}
	}


	for (kkk=0;kkk<n+1;kkk++)
	{
	  ggab[kkk]=pow(n+1-kkk,1-alpha)-pow(n-kkk,1-alpha);
	}

	for (nn=0;nn<n+1;nn++)
	{
		if (nn==0)
		{
			ggac[nn]=aabb*ggab[nn];
		}
		else 
		{
		  ggac[nn]=aabb*(ggab[nn]-ggab[nn-1]);
		}
		for (j=0;j<i2;j++)
		{
			for (ii=0;ii<j2;ii++)
			{
				tempcc[ii*i2+j]=tempcc[ii*i2+j]+ggac[nn]*Cea[nn*(i2*j2)+ii*i2+j];
			}
		}
	}


	for (j=0;j<i2;j++)
	{
		for (ii=0;ii<j2;ii++)
		{
			L[ii]=0.0;
			D[ii]=0.0;
			U[ii]=0.0;
			d[ii]=0.0;       
			Cc[ii]=C1[ii*i2+j];
			Ccmid[ii]=C1mid[ii*i2+j];
            FFc[ii]=tempcc[ii*i2+j];

			if (ii==0)
			{
				
				aab=Ccmid[ii];
				aac=Ccmid[ii+1];		

				DD =dd0*(1+(aac+aab)/2);

				R=DD*Dt/(pow(Dy,2));
				
				D[ii]=1+aabb+R;
				U[ii]=-R;
			}
			else if(ii==j2-1)
			{
				
				aaa=Ccmid[ii-1];
				aab=Ccmid[ii];

				DD=dd0*(1+(aaa+aab)/2);

			
				R  =DD*Dt/(pow(Dy,2));
				
				L[ii]=-R;
				D[ii]=1+aabb+R;

			}
			else
			{
				aaa=Ccmid[ii-1];
				aab=Ccmid[ii];
				aac=Ccmid[ii+1];

				DD0=dd0*(1+(aaa+aab)/2);

				DD1 =dd0*(1+(aac+aab)/2);
				
				R =DD0*Dt/(pow(Dy,2));
				R1=DD1*Dt/(pow(Dy,2));
				
				L[ii]=-R;
				D[ii]=1+aabb+R+R1;
				U[ii]=-R1;
			}
			d[ii]=Cc[ii]+FFc[ii]+Dt*Rightside_f(alpha,dd0,x[I1+j],y[I2+ii],(n+1)*Dt);

		}
		d[0]=d[0]+Dt/Dy*Qd[j];
		d[j2-1]=d[j2-1]-Dt/Dy*Qo[j];

		Thomas_Algorithm(L,D,U,d,j2);
              
		for(ii=0;ii<j2;ii++)  
		{
			C1mid[ii*i2+j]=d[ii];

		}
	}
	free(L);free(D);free(U);
	free(d);free(Cc);
	free(ggab);free(ggac);free(tempcc);free(FFc);

}
