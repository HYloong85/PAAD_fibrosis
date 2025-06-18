# Load required libraries for survival analysis and visualization
library(survival)
library(survminer)
library(readxl)
ptm <- proc.time()
set.seed(1)
library(gridExtra)

# Read data 
df <- read_excel("./Survdata.xlsx")
mydata = data.matrix(df)
s1 = nrow(mydata)  
s2 = ncol(mydata)
mySurv = Surv(mydata[, 4], mydata[, 5]);
x = mydata[, 7] 
data_list <- as.list(data)

# Assign patients to groups based on stroma proportion
group <- matrix(0, nrow=s1, ncol=1, byrow=T) 
T_P  <- matrix(0, nrow=61, ncol=2, byrow=T) 
nn=0
for(j in 1:s1){
  if(x[j] == 0){
    group[j] = "Low Stroma";
  }else{
    group[j] = "High Stroma";
    nn=nn+1
  }
}
# Create data frame for survival analysis
data <- data.frame(
  time = mydata[, 4],    # Time to event (months)
  status = mydata[,5],   # Event status (1=event, 0=censored)
  group = factor(        
    group,
    levels = c("Low Stroma", "High Stroma")  
  )         
)
# =====================
# Survival Analysis
# =====================

# 1. Log-rank test for group difference
log1 = survdiff(mySurv ~ group)
p = pchisq(log1$chisq, 1, lower.tail=FALSE)
print(p)

# 2. Cox proportional hazards model
cox_model <- coxph(mySurv ~ group,data=data)
hr_summary <- summary(cox_model)
hr <- hr_summary$coefficients[2]
ci <- hr_summary$conf.int[3:4]
p_hr <- hr_summary$coefficients[5]

fit <- survfit(mySurv ~ group,data=data)
p_label <- paste0("Logrank test P = ", formatC(p , format = "g", digits = 3))
hr_label <- paste0("Hazard ratio,", round(hr, 2), 
                   " (95% CI ", round(ci[1], 2), "-", round(ci[2], 2), ")")
summary(fit)$table 

# =====================
# Visualization Setup
# =====================
# Custom theme for consistent formatting
custom_theme <- function() {
  theme_survminer() %+replace%
    theme(
      plot.title = element_text(face = "bold", size = 15, hjust = 0.5),  
      axis.line.x = element_line(color = "black"),
      plot.margin = margin(t = 10, r = 30, b = 10, l = 20),
      # 风险表专用设置
      risk.table = list(
        theme = theme(
          plot.title = element_blank(),  
          axis.title.y = element_blank()
        )
      )
    )
}
# Create Kaplan-Meier plot with risk table
p1 <- ggsurvplot(fit, 
                 data = data,
                 conf.int=F,
                 # pval=p_label,
                 # pval.coord = c(70, 0.6), 
                 # # pval.size=4.5,
                 # pval.parse = TRUE,
                 legend=c(0.8, 0.8),
                 legend.labs=levels(data$group),
                 legend.title= "Type",
                 censor=F,
                 title="TCGA",
                 xlab="Time (months)",
                 ylab="Overall Survival",#Progression-Free Survival  Overall Survival
                 break.time.by = 12,
                 palette="jama",
                 risk.table=T,
                 risk.table.title="",
                 risk.table.y.text.col=T,
                 risk.table.height=.25,
                 # risk.table.y.text=FALSE,
                 risk.table.x.text =T,  
                 risk.table.y.text = T ,   
                 risk.table.col = "strata", 
                 ncensor.plot = F,
                 surv.median.line = "hv" ,
                 # ncensor.plot.height = 0.25,
                 ggtheme=custom_theme()
)
# Add hazard ratio annotation at bottom left
p1$plot <- p1$plot + 
  annotate("text",
           x = 0,  
           y = 0,           
           label = hr_label,
           hjust = 0,
           size = 4.5,          
           fontface = "bold")  
# Add p-value annotation at specified position
p1$plot <- p1$plot + 
  annotate("text",
           x =24, 
           y = 0.6,            
           label = p_label,
           hjust = 0,
           size = 4.5,          
           fontface = "bold")  

p1$table <- p1$table + 
  theme(legend.position = "none",
        axis.title.y = element_blank() ,) 
p1<- ggarrange(p1$plot, p1$table, heights = c(3, 1.2),
               ncol = 1, nrow = 2, align = "v")

p1
pdf("./KM.pdf") 
print(p1,newpage = F)
dev.off() 