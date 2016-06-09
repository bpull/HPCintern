require 'nn'
model = nn.Sequential();
inputs = 2; outputs = 1; HU1 = 50; HU2 = 50;
model:add(nn.Linear(inputs, HU1))
model:add(nn.ReLU())
model:add(nn.Linear(HU1, HU2))
model:add(nn.ReLU())
model:add(nn.Linear(HU2, outputs))

criterion = nn.MSECriterion()

local batchSize = 128
local batchInputs = torch.Tensor(batchSize, inputs)
local batchLabels = torch.DoubleTensor(batchSize)

for i=1,batchSize do
  local input = torch.randn(2)     -- normally distributed example in 2d
  local label = 1
  if input[1]*input[2]>0 then     -- calculate label for XOR function
    label = -1;
  end
  batchInputs[i]:copy(input)
  batchLabels[i] = label
end

local params, gradParams = model:getParameters()

local optimState = {learningRate=0.01}
require 'optim'

for epoch=1,50 do
  -- local function we give to optim
  -- it takes current weights as input, and outputs the loss
  -- and the gradient of the loss with respect to the weights
  -- gradParams is calculated implicitly by calling 'backward',
  -- because the model's weight and bias gradient tensors
  -- are simply views onto gradParams
  local function feval(params)
    gradParams:zero()

    local outputs = model:forward(batchInputs)
    local loss = criterion:forward(outputs, batchLabels)
    local dloss_doutput = criterion:backward(outputs, batchLabels)
    model:backward(batchInputs, dloss_doutput)

    return loss,gradParams
  end
  optim.sgd(feval, params, optimState)
end

x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(model:forward(x))
x[1] =  0.5; x[2] = -0.5; print(model:forward(x))
x[1] = -0.5; x[2] =  0.5; print(model:forward(x))
x[1] = -0.5; x[2] = -0.5; print(model:forward(x))
