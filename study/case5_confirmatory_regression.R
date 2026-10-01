# Preregistered mixed-effects tests. Requires R package lme4.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: Rscript case5_confirmatory_regression.R regression.csv output.txt")
library(lme4)
data <- read.csv(args[1])
data$base_id <- factor(data$base_id)
data$item_id <- factor(data$item_id)
data$model <- factor(data$model)
data$salience <- factor(data$salience, levels = c("implicit", "some", "explicit", "pocket_box"))
data$instruction <- factor(data$instruction, levels = c("none", "user"))
data$slot <- factor(data$slot, levels = c("initial", "received", "given"))
data$chain_length <- factor(data$chain_length)

# Primary: zero-default on all underdetermined factorial solve items. Model
# and base/item intercepts account for the matched and repeated structure.
factorial <- subset(data, kind == "factorial")
primary <- glmer(zero_default ~ salience * instruction * slot + chain_length + model +
                 (1 | base_id) + (1 | item_id), data = factorial, family = binomial,
                 control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000)))

# Targeted positive control: marked pocket/box versus omitted initial.
target <- subset(data, kind == "marked_pocket" |
                 (kind == "factorial" & slot == "initial" & salience == "implicit" & chain_length == "2"))
target$condition <- factor(ifelse(target$kind == "marked_pocket", "marked", "omitted"))
dissociation <- glmer(zero_default ~ condition * instruction + model +
                      (1 | base_id) + (1 | item_id), data = target, family = binomial,
                      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000)))

# Qwen-only scale trend: report this separately; do not fit a cross-family curve.
qwen <- subset(factorial, family == "qwen")
scale_fit <- glmer(zero_default ~ log_size * salience * instruction + slot + chain_length +
                   (1 | base_id) + (1 | item_id), data = qwen, family = binomial,
                   control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000)))

sink(args[2])
cat("Primary zero-default GLMM\n")
print(summary(primary))
cat("\nMarked versus omitted instruction GLMM\n")
print(summary(dissociation))
cat("\nQwen-only size trend GLMM\n")
print(summary(scale_fit))
cat("\nConvergence and singularity diagnostics\n")
print(primary@optinfo$conv)
print(dissociation@optinfo$conv)
print(scale_fit@optinfo$conv)
print(isSingular(primary))
print(isSingular(dissociation))
print(isSingular(scale_fit))
sink()
