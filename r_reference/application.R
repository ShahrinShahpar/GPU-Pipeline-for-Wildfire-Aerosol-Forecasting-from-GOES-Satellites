library(ggplot2)
library(tidyr)
library(bnstruct)
library(SpatialVx)
library(matrixStats)
library(MCMCpack)
library(expm)

rm(list = ls())

theme_paper <- theme(
  panel.background = element_rect(fill = "white",colour = "black", linewidth = NULL, linetype = "solid"),
  panel.grid.major = element_line(linewidth = NULL, linetype = 'solid',colour = "white"),
  panel.grid.minor = element_line(linewidth = NULL, linetype = 'solid',colour = "white"),
  plot.title = element_text(size = 18, hjust=0.5),
  axis.text = element_text(size = 14),
  axis.title=element_text(size=18),
  legend.key.size = unit(0.8, "cm"),
  legend.key.width = unit(0.8,"cm"),
  legend.key = element_rect(fill = NA),
  legend.title = element_text(color = "black", size = 18),
  legend.text = element_text(color = "black", size = 18)
)

script_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) ".")
if (is.null(script_dir) || !nzchar(script_dir)) script_dir <- "."
results_dir <- file.path(script_dir, "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
source(file.path(script_dir, 'util.R'))
load(file.path(script_dir, 'data', 'G16.aod.raw.RData'))
load(file.path(script_dir, 'data', 'G17.aod.raw.RData'))
dat.map <- map_data("county", "california")

Nr <- 60
s <- expand.grid(
  x = seq(0, 1-1/Nr, length.out = Nr),
  y = seq(0, 1-1/Nr, length.out = Nr)
)

# calculate the velocity
G16.aod.impute <- replicate(60, matrix(0, Nr, Nr))
for (i in 1:60){
  tempt <- matrix(G16.aod.raw[[i]]$AOD, Nr, Nr)
  G16.aod.impute[,,i] <- knn.impute(
    tempt,
    k = 10,
    cat.var = 1:ncol(tempt),
    to.impute = 1:nrow(tempt),
    using = 1:nrow(tempt)
  )
}
dat.velocity <- list()
N.img <- 20
for (i in 1:N.img){
  initial <- G16.aod.impute[,,i]
  final <- G16.aod.impute[,,i+1]
  of.fit <- OF(final, xhat=initial, W=15, verbose=TRUE)
  speed <- matrix(of.fit$err.mag.lin, Nr, Nr)
  angle <- matrix(of.fit$err.ang.lin, Nr, Nr)
  dat.velocity[[i]] <- data.frame(s, speed=c(speed), angle=c(angle/180*pi))
}
dat.velocity.avg.smooth <- velocity(dat.velocity, 1, 19)
v.a.x <- dat.velocity.avg.smooth$speed*cos(dat.velocity.avg.smooth$angle)
v.a.y <- dat.velocity.avg.smooth$speed*sin(dat.velocity.avg.smooth$angle)
v.a <- data.frame(v.a.x=c(v.a.x), v.a.y=c(v.a.y))

# plot the velocity in Figure 10(a)
speed <- matrix(dat.velocity.avg.smooth$speed,60,60)[seq(1,60,3), seq(1,60,3)]*60*0.04*111*12
long <- matrix(G16.aod.raw[[1]]$long,60,60)[seq(1,60,3), seq(1,60,3)]
lat <- matrix(G16.aod.raw[[1]]$lat,60,60)[seq(1,60,3), seq(1,60,3)]
angle <- matrix(dat.velocity.avg.smooth$angle,60,60)[seq(1,60,3), seq(1,60,3)]
velocity.plot <- data.frame(long=c(long),lat=c(lat),angle=c(angle),speed=c(speed))
velocity <- ggplot(velocity.plot, aes(long, lat,fill=speed,angle=angle,radius=scales::rescale(speed, c(0.05, 0.1)))) +
  geom_raster() +
  geom_spoke(arrow = arrow(length = unit(0.035, 'inches'))) +
  scale_fill_viridis_c(option = "D", limits = c(0,30)) +
  labs(fill = expression(paste('speed: ', km/h))) +
  geom_polygon(data = dat.map, aes(x = long, y = lat, group = group), inherit.aes = FALSE, fill=NA, color = "black") +
  coord_cartesian(xlim=c(-124,-121.6),ylim=c(35.0,37.4)) + 
  ggtitle('(a)') +
  theme_paper 
ggsave(file.path(results_dir, 'fig_10a.pdf'), velocity, width=15, height=11, units='cm')

# calculate the diffusivity
v.a.x = matrix(v.a$v.a.x, 60, 60)
v.a.y = matrix(v.a$v.a.y, 60, 60)
col.dif.x <- rowDiffs(v.a.x)/(1/60)
row.dif.x <- colDiffs(v.a.x)/(1/60)
col.dif.y <- rowDiffs(v.a.y)/(1/60)
row.dif.y <- colDiffs(v.a.y)/(1/60)
p1 <- cbind(col.dif.x, rep(NA,60))
p2 <- rbind(row.dif.y, rep(NA,60))
p3 <- rbind(row.dif.x, rep(NA,60))
p4 <- cbind(col.dif.y, rep(NA,60))
K <- 0.28*1/60*1/60*sqrt((p1-p2)^2+(p3+p4)^2)
K.smooth <- image.smooth(K)$z
K = matrix(K.smooth, 60, 60)
col.dif <- rowDiffs(K)/(1/60)
row.dif <- colDiffs(K)/(1/60)
D.K.x <- cbind(col.dif, col.dif[,59])
D.K.y <- rbind(row.dif, row.dif[59,])
K.ifm <- data.frame(K=c(K), D.K.x=c(D.K.x), D.K.y=c(D.K.y))

# plot the velocity in Figure 10(b)
K.plot <- data.frame(
  long=c(G16.aod.raw[[1]]$long), lat=c(G16.aod.raw[[1]]$lat),
  diffusivity=c(K.smooth*60*60*0.04*111*0.04*111*12)
)  
diffusivity <- ggplot(K.plot, aes(long, lat,fill=diffusivity)) +
  geom_raster() +
  scale_fill_viridis_c(option = "D") + 
  labs(fill = expression(paste('diffusivity: ', km^2/h))) +
  geom_polygon(data = dat.map, aes(x = long, y = lat, group = group), inherit.aes = FALSE, fill=NA, color = "black") +
  coord_cartesian(xlim=c(-124,-121.6),ylim=c(35.0,37.4)) +
  ggtitle('(b)') +
  theme_paper 
diffusivity
ggsave(file.path(results_dir, 'fig_10b.pdf'), diffusivity, width=16, height=11, units='cm')

# obtain the transition matrix G 
G_PRECALCULATION = FALSE 
# IMPORTANT NOTES: 
# IF G_PRECALCULATION IS SET AS FALSE, IT WILL REQUIRE SEVERAL HOURS FOR THE FOLLOWING LINES.
# FOR THIS REPRODUCIABLE CAPSULE, WE LET G_PRECALCULATION = TRUE AS THE DEFAULT SEETING. 
if(G_PRECALCULATION){
  load(file.path(script_dir, 'data', 'G_app.RData'))
}else{
  G_app <- expm(G_ad(1/Nr, v.a, K.ifm, Function_Omega(20)))
  save(G_app, file=file.path(script_dir, 'data', 'G_app.RData'))
}

# fit the proposed model 
N <- 20  # the highest retained frequency
Omega <- Function_Omega(N)
F <- Function_F(Nr, N, Omega)  
G17_ds <- UDS(G17.aod.raw, 25)
G16_ds <- UDS(G16.aod.raw, 25)
y1 <- list()
y2 <- list()
for (i in 1:20){
  y1[[i]] <- G17_ds[[i]]$AOD
  y2[[i]] <- G16_ds[[i]]$AOD
}
obs.ccl <- OBS_ccl2(y1, y2, F)
start = Sys.time()
set.seed(222)
fit.M2 <- Gibbs_FFBS2(obs.ccl, G_app, rep(0.1, 2*N^2), diag(0.01, 2*N^2), 10)
end = Sys.time()
print(end - start)

# plot the filtering and bias correction results in Figure 9 (row c and row d)
time_plot = c('time 5', 'time 10', 'time 15', 'time 20')
time_save = c('time5', 'time10', 'time15', 'time20')
time_id = c(5, 10, 15, 20)

fig_9_row_c <- list()
for(i in 1:4){
  tempt <- data.frame(long=G17.aod.raw[[1]]$long, lat=G17.aod.raw[[1]]$lat, AOD = F%*%fit.M2$m.flt[[time_id[i]+1]][1:N^2])
  fig_9_row_c[[i]] <- 
    ggplot(tempt, aes(long, lat, fill=AOD)) +
    geom_raster() + 
    scale_fill_viridis_c(option = "D", limits = c(NA,3.5)) + 
    theme_paper + 
    geom_polygon(data = dat.map, aes(x = long, y = lat, group = group), inherit.aes = FALSE, fill=NA, color = "black") +
    coord_cartesian(xlim=c(-124,-121.6),ylim=c(35.0,37.4)) +
    ggtitle(time_plot[i])
  file_path = file.path(results_dir, paste0("fig_9_row_c_", time_save[i], ".pdf"))
  ggsave(file_path, fig_9_row_c[[i]], width=10, height=8, units='cm')
}  

fig_9_row_d <- list()
for(i in 1:4){
  tempt <- data.frame(long=G17.aod.raw[[1]]$long, lat=G17.aod.raw[[1]]$lat, bias = F%*%fit.M2$m.flt[[time_id[i]+1]][(N^2+1):(2*N^2)])
  fig_9_row_d[[i]] <- 
    ggplot(tempt, aes(long, lat, fill=bias)) +
    geom_raster() + 
    scale_fill_viridis_c(option = "plasma", limits = c(-2.5, 2.5)) +
    theme_paper + 
    geom_polygon(data = dat.map, aes(x = long, y = lat, group = group), inherit.aes = FALSE, fill=NA, color = "black") +
    coord_cartesian(xlim=c(-124,-121.6),ylim=c(35.0,37.4)) +
    ggtitle(time_plot[i])
  file_path = file.path(results_dir, paste0("fig_9_row_d_", time_save[i], ".pdf"))
  ggsave(file_path, fig_9_row_d[[i]], width=10, height=8, units='cm')
}  

# plot the predictive differences in Figure 11 (row b and row d)
G.list <- list()
G.list[[1]] <- G_app
for(i in 2:5){
  G.list[[i]] = G_app%*%G.list[[i-1]]
}
p.G17.prd.dif <- list()
p.G16.prd.dif <- list()

for (k in 1:4) {
  data <- data.frame(
    long=G17_ds[[1]]$long,
    lat=G17_ds[[1]]$lat,
    AOD = F%*%G.list[[k]]%*%fit.M2$m.flt[[21]][1:N^2] - G17.aod.raw[[20+i]]$AOD
  )
  p.G17.prd.dif[[k]] <-
    ggplot(data, aes(long, lat, fill=AOD)) +
    geom_raster() +
    scale_fill_viridis_c(option = "plasma", limits = c(-3,3), na.value="white") +
    geom_polygon(data = dat.map, aes(x = long, y = lat, group = group), inherit.aes = FALSE, fill=NA, color = "black") +
    coord_cartesian(xlim=c(-124,-121.6),ylim=c(35.0,37.4)) +
    ggtitle(paste(c("time", as.character(k+20)), collapse=" ")) +
    theme_paper
  time = paste(c("time", as.character(k+20)), collapse="")
  file_path = file.path(results_dir, paste0("fig_11_row_b_", time, ".pdf"))
  ggsave(file_path, p.G17.prd.dif[[k]], width=10, height=8, units='cm')
}

for (k in 1:4) {
  data <- data.frame(
    long=G16_ds[[1]]$long,
    lat=G16_ds[[1]]$lat,
    AOD = F%*%G.list[[k]]%*%fit.M2$m.flt[[21]][1:N^2] - G16.aod.raw[[20+i]]$AOD
  )  
  p.G16.prd.dif[[k]] <- 
    ggplot(data, aes(long, lat, fill=AOD)) +
    geom_raster() + 
    scale_fill_viridis_c(option = "plasma", limits = c(-3,3), na.value="white") +
    geom_polygon(data = dat.map, aes(x = long, y = lat, group = group), inherit.aes = FALSE, fill=NA, color = "black") +
    coord_cartesian(xlim=c(-124,-121.6),ylim=c(35.0,37.4)) +
    ggtitle(paste(c("time", as.character(k+20)), collapse=" ")) + 
    theme_paper
  time = paste(c("time", as.character(k+20)), collapse="")
  file_path = file.path(results_dir, paste0("fig_11_row_d_", time, ".pdf"))
  ggsave(file_path, p.G16.prd.dif[[k]], width=10, height=8, units='cm')
}  

# comparison with statistical models for single source
start = Sys.time()
obs.G17.ccl <- OBS_ccl1(y1, F)
fit.G17 <- Gibbs_FFBS1(obs.G17.ccl, G_app, rep(0.1, 2*N^2), diag(0.01, 2*N^2), 10)
obs.G16.ccl <- OBS_ccl1(y2, F)
fit.G16 <- Gibbs_FFBS1(obs.G16.ccl, G_app,  rep(0.1, 2*N^2), diag(0.01, 2*N^2), 10)
end = Sys.time()
print(end-start)

dif1716 = abs(F%*%G.list[[1]]%*%fit.G16$m.flt[[21]][1:N^2] - F%*%G.list[[1]]%*%fit.G17$m.flt[[21]][1:N^2])
for (k in 2:5){
  tempt = abs(F%*%G.list[[k]]%*%fit.G16$m.flt[[21]][1:N^2] - F%*%G.list[[k]]%*%fit.G17$m.flt[[21]][1:N^2])
  dif1716 = rbind(dif1716, tempt)
}
dif16 = abs(F%*%G.list[[1]]%*%fit.G16$m.flt[[21]][1:N^2] - F%*%G.list[[1]]%*%fit.M2$m.flt[[21]][1:N^2])
for (k in 2:5){
  tempt = abs(F%*%G.list[[k]]%*%fit.G16$m.flt[[21]][1:N^2] - F%*%G.list[[k]]%*%fit.M2$m.flt[[21]][1:N^2])
  dif16 = rbind(dif16, tempt)
}
dif17 = abs(F%*%G.list[[1]]%*%fit.G17$m.flt[[21]][1:N^2] - F%*%G.list[[1]]%*%fit.M2$m.flt[[21]][1:N^2])
for (k in 2:5){
  tempt = abs(F%*%G.list[[k]]%*%fit.G17$m.flt[[21]][1:N^2] - F%*%G.list[[k]]%*%fit.M2$m.flt[[21]][1:N^2])
  dif17 = rbind(dif17, tempt)
}

# plot the absolute prediction differences in Figure 14
pdf(file=file.path(results_dir, 'fig_14a.pdf'), width=6, height=5)
hist(
  dif1716, breaks = 100, freq = FALSE,
  xlab = "prediction difference", main = '(a)',
  cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2
)
dev.off()

pdf(file=file.path(results_dir, 'fig_14b.pdf'), width=6, height=5)
hist(
  dif16, breaks = 100, freq = FALSE, 
  xlab = "prediction difference", main = '(b)',
  cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2
)
dev.off()

pdf(file=file.path(results_dir, 'fig_14c.pdf'), width=6, height=5)
hist(
  dif17, breaks = 100, freq = FALSE,
  xlab = "prediction difference", main = '(c)',
  cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2
)
dev.off()
