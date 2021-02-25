# The script for survival analysis (overall survival, progression free survival, etc) and generate KaplanMeier estimator plots
rm(list = ls())

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
	print('Arguments needed')
  	input_path = "/home/fpopp/PycharmProjects/Deepan/runs/2021-02-25/15-VarGCN/df_y.csv"
	output_path = "KM_plot.pdf"
} else if (length(args)==1) {
input_path = args[0]
output_path = args[1]
	}
# Requirement package
library(survival)

# Loading input file
# requires, sample-ID, survival time, survival events (0=alive, 1=dead) and group of sample
# an example file --> Overall_Surv_CAF_Group.txt

samples<-read.table(input_path, header=TRUE, sep="\t", dec = ".")
#kick blacked out patients by DBSCAN
samples<-samples[!(samples$labels=='-1'),]

# survival time
SURV<-samples$days_to_death
# survival events
EVENT<-samples$vital_status
# groups of samples
TYPE<-as.factor(samples$labels)

PSURV<-c()
HR <- c()
CI<- c()

# Define color of clusters
COLI<-c("red","green","royalblue","orange")

#Compute A Survival Curve For Censored Data with function survfit
BRC.bytype<-survfit(Surv(as.numeric(SURV),as.numeric(EVENT))~ TYPE)

# perform a log-rank test with function survdiff
BRC.bytype.logrank<-survdiff(Surv(as.numeric(SURV),as.numeric(EVENT))~ TYPE)
# calcualte log-rank p-value
# correct a degree of freedom of your data. 
# df=number of clusters -1

PSURV<-sprintf("%.2e",(1-pchisq(BRC.bytype.logrank[[5]], df=3)))	
n1<-as.vector(BRC.bytype.logrank[[1]])
m1<-as.vector(BRC.bytype.logrank[[2]])
o1<-paste("n = ",n1,"(",m1,")",sep="")

# Hazard ratio value
HR<-round(((BRC.bytype.logrank$obs[1]/BRC.bytype.logrank$exp[1])/(BRC.bytype.logrank$obs[2]/BRC.bytype.logrank$exp[2])),2)	
lHR1<-(BRC.bytype.logrank$obs[1]-BRC.bytype.logrank$exp[1])/(BRC.bytype.logrank$var[1,1])
HR1<-exp(1)^lHR1
V<-abs(BRC.bytype.logrank$var[1,1])

# 95% conficent interval
upHR1<-exp(1)^(lHR1+1.96/sqrt(V))
loHR1<-exp(1)^(lHR1-1.96/sqrt(V))
CI<- paste(round(loHR1,2),round(upHR1,2),sep="-")

# generate KM plot
pdf(file= output_path, width = 18, height = 15)
par(lwd=1, tcl=0.5, mar=c(7,7,5,5),yaxs = 'r')        
	plot(BRC.bytype, col=COLI, lwd=5,xlab="survival time (days)", ylab="Overall survival probability", xlim = c(0,7300), cex.lab=2.5, cex.axis=2.5)
        text(6000,0.8,paste("p=",PSURV,sep=""), col="black", cex=2)
		# If have only two group, should print out HR but if have > 2 group, consider only log rank p-value
		 text(6000,0.7,paste("HR=",HR," , CI=",CI,sep=""), col="black", cex=2)
		legend(6000, 1.0, levels(TYPE),col=COLI,bty='n', cex=2, lty=1)
        
dev.off()			
