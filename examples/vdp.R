x = seq(-2, 2, length.out = 500)
eps = 0.01

lamb1 = (-(x^2-1) - sqrt((x^2-1)^2 - 4*eps)) / (2 * eps)
lamb2 = (-(x^2-1) + sqrt((x^2-1)^2 - 4*eps)) / (2 * eps)

plot(x, lamb1, type = 'l', ylim = c(-300, 200))
lines(x, lamb2, col='red')



plot(x, abs(1 / lamb1), type = 'l', ylim = c(0, 3), main = 'characteristic time scale')
lines(x, abs(1 / lamb2), col='red')

plot(x, pmax(abs(lamb1), abs(lamb2)) / pmin(abs(lamb1), abs(lamb2)), type = 'l', ylim = c(0, 1000), main = 'time scale ratio')



