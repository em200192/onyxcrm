# training_data.py

# training_data.py

def get_training_examples():
    """
    Returns a list of triplets with titles that EXACTLY MATCH the new parser's output.
    """
    return [
        # This now exactly matches the title that will be in gl_guide_kb.json
        ("كيف اضيف سند قبض لعميل", "سند القبض / ( Receipt Voucher )", "سند الصرف / ( Payment Voucher )"),

        # This example reinforces the difference
        ("كيفية عمل سند صرف", "سند الصرف / ( Payment Voucher )", "سند القبض / ( Receipt Voucher )"),

        # This example helps distinguish between the action and the report
        ("كيف اضيف سند قبض", "سند القبض / ( Receipt Voucher )", "تقارير سند القبض / ( Receipt Voucher Reports )"),

        # This example helps with the 'Banks' vs 'Bank Groups' confusion
        ("كيف افتح بنك", "البنوك / ( Banks )", "مجموعات البنوك / ( Bank Groups )"),

    ]