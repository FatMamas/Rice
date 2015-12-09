# get data
args <- commandArgs(trailingOnly = TRUE)
file.name <- args[1]
chunks <- unlist(strsplit(file.name, split="[/.]"))
name <- head(tail(chunks, 2), 1)
data <- read.csv(file.name, sep=';')

# begin plotting
png(file = paste(name, "png", sep='.'), bg = "white", width=1500, height=850)
par(mfrow=c(1, 2), oma = c(0, 0, 2, 0))

# Valacc
plot(data$epoch, data$valacc, type='l', main='Validation accuracy', xlab='Epoch [#]', ylab='Accuracy [%]')
abline(a=max(data$valacc), b=0, col='red')
axis(side = 2, at = max(data$valacc))

#Tloss
plot(data$epoch, data$trainloss, type='l', main='Train/Test loss', xlab='Epoch [#]', ylab='Loss')
lines(data$epoch, data$valloss, col='red')
legend('topright',
       c('Train loss', 'Test loss'),
       col=c('black', 'red'),
       lty = 1, bty='n')

# end of plots
mtext(paste('Summary for', name), outer = TRUE, cex = 1.5)
par(mfrow=c(1,1))
dev.off()
# Print summary
summary(data)
