# ==============================================================================
# 文件名: beginner_friendly_solver.jl
# 目标: 找出三次方程 ax^3 + bx^2 + cx + d = 0 的所有解 (a 不能等于 0)
# 核心方法: 待办事项本 + 切分盒子(B&P) + 魔法缩小(区间牛顿法)
# ==============================================================================

using IntervalArithmetic

# ------------------------------------------------------------------------------
# 第一步：准备最基础的数学公式计算工具 (不使用任何复杂结构体)
# ------------------------------------------------------------------------------

# 1. 计算原方程的结果: g(x) = ax^3 + bx^2 + cx + d
# 只要你给我一个 x (可以是一个具体的数，也可以是一个区间盒子)，我就按公式算出来
function calculate_equation(x, a, b, c, d)
    return a * x^3 + b * x^2 + c * x + d
end

# 2. 计算方程的导数（斜率）: g'(x) = 3ax^2 + 2bx + c
# 这是为了给牛顿法提供支持
function calculate_derivative(x, a, b, c, d)
    return 3 * a * x^2 + 2 * b * x + c
end


# ------------------------------------------------------------------------------
# 第二步：编写大白话版本的核心搜索程序
# ------------------------------------------------------------------------------

function start_searching(a, b, c, d, start_min, start_max, stop_width)
    println("====== 搜索任务开始 ======")
    println("正在解的方程是: $(a)x^3 + $(b)x^2 + $(c)x + $(d) = 0")
    
    # 把普通的数字边界，转换成 IntervalArithmetic 认识的“区间盒子”
    # “..” 是这个库特有的符号，表示从左边到右边的一个范围
    initial_box = start_min .. start_max
    
    # 建立一个“成功档案袋”，用来装最后找到的极小盒子
    success_results = []
    
    # 建立一个“待办事项本”，把最一开始的大盒子写进去
    # 用数组表示，就像排队一样
    todo_list = [initial_box]
    
    # 只要待办事项本上还有没查过的盒子，就一直循环查下去
    while length(todo_list) > 0
        
        # 1. 从待办事项本的最后拿出一个盒子来检查，并从本子上划掉它
        current_box = pop!(todo_list)
        
        # 2. 检查这个盒子里到底有没有 0 
        # 把这个盒子代入方程，看看算出来的结果范围
        y_result = calculate_equation(current_box, a, b, c, d)
        
        # in(0, y_result) 的意思是：数字 0 是否在这个结果范围里面？
        # 如果前面加个 !，就表示“如果 0 不在里面”
        if !in(0, y_result)
            # 钥匙绝对不可能在这里，直接跳过后面的步骤，去拿下一个盒子
            continue 
        end
        
        # 3. 检查盒子是不是已经足够小了？
        # diam(current_box) 是计算这个盒子的宽度 (上限减去下限)
        if diam(current_box) <= stop_width
            # 盒子已经极小了，说明我们精确锁定了根的位置！
            push!(success_results, current_box)
            println("【找到一个根！】位置在: ", current_box)
            # 存起来之后，这个盒子就查完了，去查下一个
            continue 
        end
        
        # 4. 如果盒子还比较大，我们尝试用“牛顿法魔法”来大幅度缩小它
        # 先算出这个盒子的导数范围
        deriv_result = calculate_derivative(current_box, a, b, c, d)
        
        # 牛顿法有一个致命弱点：导数范围里绝对不能有 0 (因为后面要做除法)
        # 所以我们先检查，如果 0 不在导数范围里，才可以使用牛顿法
        if !in(0, deriv_result)
            
            # 第一步：找出这个盒子最中间的那个点
            center_point = mid(current_box)
            
            # 第二步：算出中点对应的具体函数值
            value_at_center = calculate_equation(center_point, a, b, c, d)
            
            # 第三步：套用牛顿法公式算出新的范围
            # 公式：新范围 = 中点 - (中点的值 / 导数范围)
            newton_box = center_point - (value_at_center / deriv_result)
            
            # 第四步：取交集
            # ∩ 符号表示取两个盒子的重合部分。真实的解一定在这个重合部分里
            shrunk_box = current_box ∩ newton_box
            
            # 如果重合部分是空的 (isempty)，说明里面根本没根
            if isempty(shrunk_box)
                continue
            end
            
            # 如果牛顿法很给力，成功把盒子缩小了，我们就把缩小后的盒子重新放回待办事项本
            if diam(shrunk_box) < diam(current_box)
                push!(todo_list, shrunk_box)
                # 牛顿法成功了，当前步骤结束，进入下一轮循环
                continue 
            end
        end
        
        # 5. 如果牛顿法用不了（或者没怎么缩小），我们就只能用“切开”的笨办法
        # 找到当前盒子的中点
        cut_point = mid(current_box)
        
        # 用 inf() 获取下限，sup() 获取上限，把大盒子切成左右两个小盒子
        left_box = inf(current_box) .. cut_point
        right_box = cut_point .. sup(current_box)
        
        # 把切好的两个新盒子写回待办事项本
        push!(todo_list, right_box)
        push!(todo_list, left_box)
        
    end # 循环结束的标记
    
    # 把成功档案袋里的结果交出去
    return success_results
end


# ------------------------------------------------------------------------------
# 第三步：设定参数并运行测试
# ------------------------------------------------------------------------------

# 设定三次方程的四个参数 (对应 ax^3 + bx^2 + cx + d = 0)
# 我们用 x^3 - 6x^2 + 11x - 6 = 0 这个方程来测试，它的根我们都知道是 1, 2, 3
A = 1.0
B = -6.0
C = 11.0
D = -6.0

# 设定初始的大致搜索范围: 从 -10 搜到 10
start_point = -10.0
end_point = 10.0

# 设定盒子的目标宽度：当盒子宽度小于 0.00001 时，我们认为足够精确了
target_width = 0.00001

# 调用我们上面写的搜索函数，开始执行！
final_roots = start_searching(A, B, C, D, start_point, end_point, target_width)

println("\n====== 最终报告 ======")
println("一共找到了 ", length(final_roots), " 个解。")
for (index, root_box) in enumerate(final_roots)
    println("第 ", index, " 个解所在的极小范围是: ", root_box)
    println("    -> 约等于具体数值: ", mid(root_box))
end
