# mix x^2

beta1 = 0.9
beta2 = 0.99
eps   = 1e-8
alpha = 0.001

v = zeros(100)
m = zeros(100)
x = zeros(100)

x[1] = 10

for i = 1:99
    g      = 2*x[i]
    m[i+1] = beta1*m[i] + (1-beta1)*g
    v[i+1] = beta2*v[i] + (1-beta2)*g^2
    mhat   = m[i+1]/(1-beta1^i)
    vhat   = v[i+1]/(1-beta2^i)
    x[i+1] = x[i] - alpha*mhat/sqrt(vhat + eps)
    println(x[i+1])
end

