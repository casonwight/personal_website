library(splines)
library(nlme)
library(ggplot2)
library(geoR)
library(gridExtra)
library(GGally)
library(ggpubr)
source("predictgls.R")
source("stdres.gls.R")


# Defining a ggplot funtion to get added variable plots
avPlots.invis <- function(MODEL, ...) {
  
  ff <- tempfile();
  png(filename = ff);
  OUT <- car::avPlots(MODEL, ...);
  dev.off()
  unlink(ff);
  OUT; }

ggAVPLOTS  <- function(MODEL, YLAB = NULL) {
  
  #Extract the information for AV plots
  AVPLOTS <- avPlots.invis(MODEL)
  K       <- length(AVPLOTS)
  
  #Create the added variable plots using ggplot
  GGPLOTS <- vector('list', K)
  for (i in 1:K) {
    DATA         <- data.frame(AVPLOTS[[i]]);
    GGPLOTS[[i]] <- ggplot2::ggplot(aes_string(x = colnames(DATA)[1], 
                                               y = colnames(DATA)[2]), 
                                    data = DATA) +
      geom_point(colour = 'blue') + 
      geom_smooth(method = 'lm', se = FALSE, 
                  color = 'red', formula = y ~ x, linetype = 'dashed') +
      xlab(paste0('Predictor Residual \n (', 
                  names(DATA)[1], ' | others)')) +
      ylab(paste0('Response Residual \n (',
                  ifelse(is.null(YLAB), 
                         paste0(names(DATA)[2], ' | others'), YLAB), ')')) + 
      theme(axis.text=element_text(size=5)); }
  
  #Return output object
  GGPLOTS; }

# Reading in the data
forest <- read.csv('LodgepoleInUintas.csv', header = TRUE)

forest.train <- forest[!is.na(forest$Lodgepole),]
forest.test <- forest[is.na(forest$Lodgepole),]

ggpairs(forest.train[,3:6]) + ggtitle("Scatterplots of Lodgepole and Covariates")

plot_usmap(include = "UT") + 
  geom_point(data = usmap_transform(forest.train),
             aes(x=LON.1, y = LAT.1, color = Lodgepole)) +
  labs(title = "Basal Area of Lodgepole Pines") + 
  theme(legend.position = "right")



# Showing the nonlinearity of Elevation
plots <- ggAVPLOTS(lm(Lodgepole ~ LON + LAT + Slope + Aspect + ELEV, data = forest.train))
plots[[5]]+ggtitle("Added Variable Plot for Elevation")
plots[[4]]+ggtitle("Added Variable Plot for Aspect")


data <- data.frame(x = cos(forest.train$Aspect*(2*pi/360+pi/2)), y = sin(forest.train$Aspect*(2*pi/360+pi/2)), Lodgepole = forest.train$Lodgepole)
ggplot(data, aes(x=x, y = y, color = Lodgepole)) +
  geom_point() + 
  ggtitle("Scatterplot of Aspect") + 
  scale_color_gradient(low="blue", high="red") +
  annotate("text", label = c("North", "West", "South", "East"), y = c(.85,0,-.85,0), x = c(0,-.85,0,.85), cex = 4.5)


grid.arrange(grobs = list(plots[[4]],plots[[5]]), top = "Added Variable Plots for Aspect")
data <- data.frame("Elevation" = forest.train$ELEV, "Lodgepole" = forest.train$Lodgepole)
ggplot(data, aes(x=Elevation, y = Lodgepole)) +
  geom_point() + 
  ggtitle("Scatterplot of Lodgepole Basal Area and Elevation")

# Possible knots for elevation
ELEV.knots <- 1:10

BICs <- c()
AICs <- c()

for(i in ELEV.knots){
  lm <- lm(Lodgepole ~ LON + LAT + Slope + sin(Aspect*(2*pi/360+pi/2)) + cos(Aspect*(2*pi/360+pi/2)) + ns(ELEV, df = i), data = forest.train)
  BICs[i] <- BIC(lm)
  AICs[i] <- AIC(lm)
}

# Best number of knots (4)
AIC.ELEVknot <- which.min(AICs)

groups <- 1:10
forest.train$Group <- sample(1:length(groups), nrow(forest.train), replace = TRUE)
RMSE <- c()

final.lm <- lm(Lodgepole ~ LON + LAT + Slope + sin(Aspect*(2*pi/360+pi/2)) + cos(Aspect*(2*pi/360+pi/2)) + ns(ELEV, df = AIC.ELEVknot), data = forest.train)

forest.varioG <- gstat::variogram(Lodgepole ~ Slope + sin(Aspect*(2*pi/360+pi/2)) + cos(Aspect*(2*pi/360+pi/2)) + ns(ELEV, df = 4), loc= ~ LON + LAT, data = forest.train)

ggplot(forest.varioG, aes(x=dist, y=gamma)) + 
  geom_point() + 
  ggtitle("Semivariogram") + 
  xlab("Distance") +
  ylab("Semivariance")


# Get the RMSE
for(group in groups){
  outies <- forest.train[forest.train$Group!=group,]
  groupies <- forest.train[forest.train$Group==group,]
  exp.gls <- gls(Lodgepole ~Slope + sin(Aspect*(2*pi/360+pi/2)) + cos(Aspect*(2*pi/360+pi/2)) + ns(ELEV, df = 4),
                correlation=corExp(form = ~LON + LAT,nugget=TRUE), data=outies, method="ML")
  pred <- predictgls(exp.gls, groupies)$Prediction
  RMSE[group] <- sqrt(mean((pred-groupies$Lodgepole)^2))
}
mean(RMSE)
sd(forest.train$Lodgepole)

final.gls <- gls(Lodgepole ~Slope + sin(Aspect*(2*pi/360+pi/2)) + cos(Aspect*(2*pi/360+pi/2)) + ns(ELEV, df = 4),
                correlation=corExp(form = ~LON + LAT,nugget=TRUE), data=forest.train, method="ML")
predictions <- predictgls(final.gls, forest.test)
preds <- predictions$Prediction
lwr.ci <- predictions$lwr
upr.ci <- predictions$upr

# R^2
R2 <- cor(forest.train$Lodgepole, predictgls(final.gls, forest.train)$Prediction)^2
R2

# Linearity - AV plots
grid.arrange(grobs = ggAVPLOTS(final.lm), top = "Added Variable Plots")

# Normality - Density of the decorrelated residuals
# Testing to see if decorrelated residuals are non-normal
p.val <- round(ks.test(stdres.gls(final.gls), "pnorm",0,1)$p.val,4)

data <- data.frame("Residuals" = stdres.gls(final.gls))
ggplot(data = data, aes(x=Residuals)) + 
  geom_density(aes(colour = "Decorrelated Residuals"), lwd = 1.05) + 
  stat_function(aes(colour = "Standard Normal"), fun = function(x) dnorm(x,0,1), lwd = 1.05) +
  xlab("Decorrelated Residuals") +
  ylab("Density") +
  ggtitle("Density Plot of the Decorrelated Residuals") + 
  scale_colour_manual("Density", values = c("royalblue", "red")) + 
  annotate("text", label = paste("KS Test p-value:\n", p.val), x = 3, y = .45, cex = 4.5)
  


# Equal Variance - Residuals vs Fitted values (removed number 35)
predictions <- predictgls(final.gls, forest.train)$Prediction
data <- data.frame(x = predictions, y = stdres.gls(final.gls))
ggplot(data = data, aes(x=x, y=y)) + 
  geom_point() + 
  annotate("text", label = "Max Response", x = 14.5, y = 5.1) + 
  xlab("Fitted Values") +
  ylab("Residuals") + 
  ggtitle("Fitted Values vs. Standardized Residuals") + 
  geom_hline(aes(colour = "red", yintercept = 0), show.legend = FALSE)


# Testing to see if Elevation has a nonlinear effect
final.gls.no.spline <- gls(Lodgepole ~ Slope + sin(Aspect*(2*pi/360+pi/2)) + cos(Aspect*(2*pi/360+pi/2)) + ELEV, 
                 correlation=corSpatial(form = ~LON + LAT,nugget=TRUE, 
                                        type = "gaussian"), data=forest.train, method="ML")
anova(final.gls, final.gls.no.spline)




final.gls.no.ELEV <- gls(Lodgepole ~ Slope + sin(Aspect*(2*pi/360+pi/2)) + cos(Aspect*(2*pi/360+pi/2)), 
                           correlation=corSpatial(form = ~LON + LAT,nugget=TRUE, 
                                                  type = "gaussian"), data=forest.train, method="ML")
fitted.no.ELEV  <- predictgls(final.gls.no.ELEV, forest.train)$Prediction

fitted <- predictgls(final.gls, forest.train)$Prediction

ELEV.res <- data.frame("Elevation" = forest.train$ELEV, "Residuals" = fitted-fitted.no.ELEV)

Elevation.index <- seq(0,max(ELEV.res$Elevation), by = 1)

elevation.lm <- lm(Residuals ~ ns(Elevation, df = 4), data = ELEV.res)

ELEV.fit <- data.frame("Elevation" = Elevation.index, "Predictions" = predict(elevation.lm, data.frame("Elevation" = Elevation.index)))


final.lm <- lm(Lodgepole ~ Slope + sin(Aspect*(2*pi/360+pi/2)) + cos(Aspect*(2*pi/360+pi/2)) + ns(ELEV, df = 4), data = forest.train)
resids <- as.numeric(model.matrix(final.lm)%*%matrix(coef(final.gls),ncol = 1)) - as.numeric(model.matrix(final.lm)[,-(5:8)]%*%matrix(coef(final.gls)[-(5:8)],ncol = 1))
act.resids <- as.numeric(forest.train$Lodgepole- as.numeric(model.matrix(final.lm)[,-(5:8)]%*%matrix(coef(final.gls)[-(5:8)],ncol = 1)))

resids.low <- as.numeric(model.matrix(final.lm)%*%matrix(coef(final.gls),ncol = 1)) - as.numeric(model.matrix(final.lm)[,-(5:8)]%*%matrix(coef(final.gls)[-(5:8)],ncol = 1)) -
  as.numeric(qt(.975,nrow(forest.train)-length(coef(final.gls))-2)*sqrt(diag(model.matrix(final.lm)[,(5:8)]%*%vcov(final.gls)[(5:8),(5:8)]%*%t(model.matrix(final.lm)[,(5:8)]))), ncol = 1)
resids.upp <- as.numeric(model.matrix(final.lm)%*%matrix(coef(final.gls),ncol = 1)) - as.numeric(model.matrix(final.lm)[,-(5:8)]%*%matrix(coef(final.gls)[-(5:8)],ncol = 1)) +
  as.numeric(qt(.975,nrow(forest.train)-length(coef(final.gls))-2)*sqrt(diag(model.matrix(final.lm)[,(5:8)]%*%vcov(final.gls)[(5:8),(5:8)]%*%t(model.matrix(final.lm)[,(5:8)]))), ncol = 1)

resss <- c()
for(i in 1:1000){
  this.train <- forest.train[sample(nrow(forest.train), replace = TRUE),]
  this.lm <- lm(Lodgepole ~ Slope + sin(Aspect*(2*pi/360+pi/2)) + cos(Aspect*(2*pi/360+pi/2)) + ns(ELEV, df = 4), data = this.train)
  resss <- cbind(resss,predict(this.lm, forest.train) - as.numeric(model.matrix(final.lm)[,-(5:8)]%*%matrix(coef(this.lm)[-(5:8)],ncol = 1)))
}

resids.low <- apply(resss,1,quantile,.005)
resids.upp <- apply(resss,1,quantile,.995)

ELEV.res <- data.frame("Elevation" = forest.train$ELEV, "Residuals" = resids)

ggplot(ELEV.res, aes(x=Elevation, y=Residuals)) + 
  geom_line(color = "red") +
  geom_line(aes(x=Elevation, y = resids.low), color = "red", lty = "dashed") +
  geom_line(aes(x=Elevation, y = resids.upp), color = "red", lty = "dashed") +
  geom_point(aes(x=Elevation, y = act.resids)) +
  ylab(expression(paste(hat(y)-X[0],beta[0]))) +
  xlab("Elevation") +
  ggtitle("Nonlinear Effect of Elevation on Lodgepole Basal Area")



new <- data.frame("Aspect" = seq(0,360,by =.1))
new$x <- cos(new$Aspect*2*pi/360+pi/2)
new$y <- sin(new$Aspect*2*pi/360+pi/2)
new$Effect <- new$y*final.gls$coefficients[3] + new$x*final.gls$coefficients[4]


ggplot(new, aes(x = x, y = y, color = Effect)) + 
  geom_point() + 
  ggtitle("Effect of Aspect on Lodgepole Basal Area") +
  annotate("text", label = c("North", "West", "South", "East"), y = c(.85,0,-.85,0), x = c(0,-.85,0,.85), cex = 4.5) + 
  scale_color_gradient(low="blue", high="red")


plot1 <- ggplot(forest.train, aes(x = LON, y = LAT, color = Lodgepole)) + 
  geom_point() + 
  scale_color_gradient(low="blue", high="red")

predictions <- predictgls(final.gls, forest.test)$Prediction
data <- cbind(forest.test, predictions)
plot2 <- ggplot(data, aes(x = LON, y = LAT, color = predictions)) + 
  geom_point() + 
  scale_color_gradient(low="blue", high="red")

grid.arrange(grobs = list(plot1 + ggtitle("Actual Lodepole Basal Area"),plot2 + ggtitle("Predicted Lodepole Basal Area")), nrow = 1)



ggarrange(plot1 + ggtitle("Actual Lodepole Basal Area"), plot2 + ggtitle("Predicted Lodepole Basal Area"), common.legend = FALSE, nrow = 1, ncol = 2, legend = "right")


data <- cbind(forest.test, "lwr"= predictgls(final.gls, forest.test)$lwr)
plot3 <- ggplot(data, aes(x = LON, y = LAT, color = lwr)) + 
  geom_point() + 
  scale_color_gradient(low="blue", high="red") + 
  ggtitle("95% Lower Predicted Lodepole Basal Area")

data <- cbind(forest.test, "upr" = predictgls(final.gls, forest.test)$upr)
plot4 <- ggplot(data, aes(x = LON, y = LAT, color = upr)) + 
  geom_point() + 
  scale_color_gradient(low="blue", high="red") + 
  ggtitle("95% Upper Predicted Lodepole Basal Area")
ggarrange(plot3, plot4, common.legend = FALSE, nrow = 1, ncol = 2, legend = "right")


betahat <- matrix(coef(final.gls), ncol = 1)

lwr <- c()
upr <- c()
for(i in 1:nrow(betahat)){
  x <- matrix(rep(0,nrow(betahat)), nrow = 1)
  x[1,i] <- 1
  lwr[i] <- as.numeric(x%*%betahat - qt(.975,111)*sqrt(diag(x%*%vcov(final.gls)%*%t(x))))
  upr[i] <- as.numeric(x%*%betahat + qt(.975,111)*sqrt(diag(x%*%vcov(final.gls)%*%t(x))))
}
betahat
lwr
upr

vcov(final.gls)



