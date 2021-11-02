from copy import deepcopy


# An expression tree node
class ExpressionTreeNode(object):
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.left = None
        self.right = None

    def is_none(self):
        return self.value == None

    def is_left_none(self):
        return self.left == None

    def is_right_none(self):
        return self.right == None


class ExpressionTree(object):
    def __init__(self):
        self.ops_list = [';', 'SEP', '=', '+', '-', '*', '/', '^']
        self.ops_priority = {";": 0, "SEP": 0, "=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "^": 4}
        self.root = None

    def build_tree_from_infix_expression(self, infix_expression):  # expression -- equation word list
        new_infix_expression = []
        exp_len = len(infix_expression)
        idx = 0
        while idx < exp_len:
            if idx == 0 and infix_expression[idx] == '-':
                new_infix_expression.append('-'+infix_expression[idx + 1])
                idx = idx + 2
            elif idx + 1 < exp_len and infix_expression[idx] == '-' and (infix_expression[idx - 1] in self.ops_list or infix_expression[idx - 1] == '('):
                new_infix_expression.append('-'+infix_expression[idx + 1])
                idx = idx + 2
            else:
                new_infix_expression.append(infix_expression[idx])
                idx = idx + 1
        infix_expression = new_infix_expression

        postfix_expression = self._infix2postfix(infix_expression)

        et_stack = []
        # print(postfix_expression)
        # 遍历后缀表达式的每个元素
        for elem in postfix_expression:
            # if operand, simply push into stack
            if elem not in self.ops_list: #["+", "-", "*", "/", "^", '=']:
                if elem[0] == '-':
                    new_node = ExpressionTreeNode('-')
                    new_node.right = ExpressionTreeNode(elem[1:])
                else:
                    new_node = ExpressionTreeNode(elem)
                et_stack.append(new_node)
            else:
                # Operator
                # Pop two top nodes in stack, make them as children, and add this subexpression to stack

                new_node = ExpressionTreeNode(elem)
                right_node = None
                if len(et_stack) > 0:
                    right_node = et_stack.pop()
                left_node = None
                if len(et_stack) > 0:
                    left_node = et_stack.pop()
                if right_node:
                    right_node.parent = new_node
                if left_node:
                    left_node.parent = new_node

                if right_node:
                    new_node.right = right_node
                if left_node:
                    new_node.left = left_node

                et_stack.append(new_node)

        # Only element  will be the root of expression tree
        if len(et_stack) == 0:
            self.root = None
        else:
            self.root = et_stack.pop()

    def build_tree_from_postfix_expression(self, postfix_expression):
        et_stack = []
        # print(postfix_expression)
        # 遍历后缀表达式的每个元素
        for elem in postfix_expression:
            # if operand, simply push into stack
            if elem in ['(',')','[',']']:
                continue

            if elem not in self.ops_list: #["+", "-", "*", "/", "^", '=']:
                new_node = ExpressionTreeNode(elem)
                et_stack.append(new_node)
            else:
                # Operator
                # Pop two top nodes in stack, make them as children, and add this subexpression to stack
                new_node = ExpressionTreeNode(elem)
                # right_node = et_stack.pop()
                # left_node = None
                # if len(et_stack) > 0:
                #     left_node = et_stack.pop()
                #
                # right_node.parent = new_node
                # if left_node:
                #     left_node.parent = new_node
                #
                # new_node.right = right_node
                # if left_node:
                #     new_node.left = left_node
                right_node = None
                if len(et_stack) > 0:
                    right_node = et_stack.pop()
                left_node = None
                if len(et_stack) > 0:
                    left_node = et_stack.pop()

                if right_node:
                    right_node.parent = new_node
                if left_node:
                    left_node.parent = new_node

                if right_node:
                    new_node.right = right_node
                if left_node:
                    new_node.left = left_node


                et_stack.append(new_node)

        # Only element  will be the root of expression tree
        self.root = et_stack.pop()

    def build_tree_from_prefix_expression(self, prefix_expression):
        # postfix_expression = deepcopy(prefix_expression)
        # postfix_expression.reverse()
        # self.build_tree_from_postfix_expression(postfix_expression)
        et_stack = []
        # print(postfix_expression)
        # 遍历后缀表达式的每个元素
        prefix_expression.reverse()
        # new_prefix_expression = []
        # idx = 0
        # while idx < len(prefix_expression):
        #     if idx - 1 >=0 and idx + 1 < len(prefix_expression):
        #         if prefix_expression[idx - 1] not in ["+", "-", "*", "/", "^", "="] and \
        #                         prefix_expression[idx] == '-' and \
        #                         prefix_expression[idx + 1] not in ["+", "-", "*", "/", "^", "="]:
        #             new_prefix_expression[-1] = '-' + new_prefix_expression[-1]
        #         else:
        #             new_prefix_expression.append(prefix_expression[idx])
        #     else:
        #         new_prefix_expression.append(prefix_expression[idx])
        #     idx = idx + 1
        # prefix_expression = new_prefix_expression

        for elem in prefix_expression:
            # if operand, simply push into stack
            if elem in ['(',')','[',']']:
                continue

            if elem not in self.ops_list: #["+", "-", "*", "/", "^", '=']:
                if elem[0] == '-':
                    new_node = ExpressionTreeNode('-')
                    new_node.right = ExpressionTreeNode(elem[1:])
                else:
                    new_node = ExpressionTreeNode(elem)
                et_stack.append(new_node)
            else:
                # Operator
                # Pop two top nodes in stack, make them as children, and add this subexpression to stack
                new_node = ExpressionTreeNode(elem)
                if elem == '-' and len(et_stack) == 1:
                    right_node = et_stack.pop()
                    left_node = None
                    right_node.parent = new_node
                    new_node.left = left_node
                    new_node.right = right_node
                else:
                    left_node = None
                    if len(et_stack) > 0:
                        left_node = et_stack.pop()
                    right_node = None
                    if len(et_stack) > 0:
                        right_node = et_stack.pop()

                    if left_node:
                        left_node.parent = new_node
                    if right_node:
                        right_node.parent = new_node

                    if left_node:
                        new_node.left = left_node
                    if right_node:
                        new_node.right = right_node

                et_stack.append(new_node)

        # Only element  will be the root of expression tree
        if len(et_stack) == 0:
            self.root = None
        else:
            self.root = et_stack.pop()

    def get_infix_expression(self):
        return self._infix(self.root)

    def get_prefix_expression(self):
        return self._prefix(self.root)

    def get_postfix_expression(self):
        return self._postfix(self.root)

    def _infix(self, et_node):
        if et_node == None:
            return []
        else:
            if et_node.value in self.ops_list: #["+", "-", "*", "/", "^", '=']:
                # if et_node.value == '^':
                #     left_list = self._infix(et_node.left)
                #     right_list = self._infix(et_node.right)
                # else:
                if et_node.left and et_node.left.value in ["+", "-", "*", "/", "=","^"] and self.ops_priority[et_node.value] > self.ops_priority[et_node.left.value]:
                    left_list = ['('] + self._infix(et_node.left) + [')']
                else:
                    left_list = self._infix(et_node.left)

                # if et_node.right and et_node.value == "-" and et_node.right.value == "+":
                if et_node.right and et_node.right.value in ["+", "-", "*", "/", "=","^"] and self.ops_priority[et_node.value] >= self.ops_priority[et_node.right.value]:
                    right_list = ['('] + self._infix(et_node.right) + [')']
                else:
                    right_list = self._infix(et_node.right)
                return left_list + [et_node.value] + right_list
            else:
                return self._infix(et_node.left) + [et_node.value] + self._infix(et_node.right)

    def _prefix(self, et_node):
        if et_node == None:
            return []
        else:
            if et_node.value in self.ops_list:
                # if et_node.value == '^':
                #     left_list = self._prefix(et_node.left)
                #     right_list = self._prefix(et_node.right)
                # else:
                # if et_node.left and et_node.left.value in ["+", "-", "*", "/", "=","^"] and self.ops_priority[et_node.value] > self.ops_priority[et_node.left.value]:
                #     left_list = ['('] + self._prefix(et_node.left) + [')']
                # else:
                #     left_list = self._prefix(et_node.left)

                # if et_node.right and et_node.value == "-" and et_node.right.value == "+":
                # if et_node.right and et_node.right.value in ["+", "-", "*", "/", "=","^"] and self.ops_priority[et_node.value] >= self.ops_priority[et_node.right.value]:
                #     right_list = ['('] + self._prefix(et_node.right) + [')']
                # else:
                #     right_list = self._prefix(et_node.right)
                left_list = self._prefix(et_node.left)
                right_list = self._prefix(et_node.right)
                return [et_node.value] + left_list + right_list
            else:
                return [et_node.value] + self._prefix(et_node.left) + self._prefix(et_node.right)

    def _postfix(self, et_node):
        if et_node == None:
            return []
        else:
            if et_node.value in self.ops_list:
                # if et_node.value == '^':
                #     left_list = self._postfix(et_node.left)
                #     right_list = self._postfix(et_node.right)
                # else:
                # if et_node.left and et_node.left.value in ["+", "-", "*", "/", "=","^"] and self.ops_priority[et_node.value] > self.ops_priority[et_node.left.value]:
                #     left_list = ['('] + self._postfix(et_node.left) + [')']
                # else:
                #     left_list = self._postfix(et_node.left)

                # if et_node.right and et_node.value == "-" and et_node.right.value == "+":
                # if et_node.right and et_node.right.value in ["+", "-", "*", "/", "=","^"] and self.ops_priority[et_node.value] >= self.ops_priority[et_node.right.value]:
                #     right_list = ['('] + self._postfix(et_node.right) + [')']
                # else:
                #     right_list = self._postfix(et_node.right)
                left_list = self._postfix(et_node.left)
                right_list = self._postfix(et_node.right)
                return left_list + right_list + [et_node.value]
            else:
                return self._postfix(et_node.left) + self._postfix(et_node.right) + [et_node.value]

    def _infix2postfix(self, infix_expression):
        ops_stack = list()
        postfix_expression = list()
        for elem in infix_expression:
            if elem in ['(','[']:
                ops_stack.append(elem)
            elif elem == ')':
                if len(ops_stack) == 0:
                    continue
                ops = ops_stack.pop()
                while ops != '(':
                    postfix_expression.append(ops)
                    if len(ops_stack) > 0:
                        ops = ops_stack.pop()
                    else:
                        break
            elif elem == ']':
                if len(ops_stack) == 0:
                    continue
                ops = ops_stack.pop()
                while ops != '[':
                    postfix_expression.append(ops)
                    if len(ops_stack) > 0:
                        ops = ops_stack.pop()
                    else:
                        break
            elif elem in self.ops_priority:
                while len(ops_stack) > 0 and ops_stack[-1] not in ['(','['] and \
                                self.ops_priority[elem] <= self.ops_priority[ops_stack[-1]]:
                    # 这说明当前操作的顺序比ops_stack的操作顺序优先级低
                    # 先将ops_stack中优先级比当前操作符高的放进后缀表达式
                    # 直到第一个低于当前优先级的操作符为止
                    postfix_expression.append(ops_stack.pop())
                ops_stack.append(elem)
            else:
                # 操作数直接放入后缀表达式中
                postfix_expression.append(elem)
        while len(ops_stack) > 0:
            postfix_expression.append(ops_stack.pop())
        return postfix_expression

    def _infix2prefix(self, infix_expression):
        ops_stack = list()
        prefix_expression = list()
        reverse_infix_expression = deepcopy(infix_expression).reverse()
        reverse_infix_expression.reverse()
        for elem in reverse_infix_expression:
            if elem in [")", "]"]:
                ops_stack.append(elem)
            elif elem == "(":
                ops = ops_stack.pop()
                while ops != ')':
                    prefix_expression.append(ops)
                    ops = ops_stack.pop()
            elif elem == "[":
                ops = ops_stack.pop()
                while ops != ']':
                    prefix_expression.append(ops)
                    ops = ops_stack.pop()
            elif elem in self.ops_priority:
                while len(ops_stack) > 0 and ops_stack[-1] not in ['(','['] and \
                                self.ops_priority[elem] < self.ops_priority[ops_stack[-1]]:
                    # 这说明当前操作的顺序比ops_stack的操作顺序优先级低
                    # 先将ops_stack中优先级比当前操作符高的放进前缀表达式
                    # 直到第一个低于当前优先级的操作符为止
                    prefix_expression.append(ops_stack.pop())
                ops_stack.append(elem)
            else:
                # 操作数直接放入前缀表达式中
                prefix_expression.append(elem)
        while len(ops_stack) > 0:
            prefix_expression.append(ops_stack.pop())
        prefix_expression = prefix_expression.reverse()
        return prefix_expression
