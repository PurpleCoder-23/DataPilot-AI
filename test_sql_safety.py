from agents.sql_analyst import validate_read_only_sql


test_cases = [
    # Safe queries
    ("Simple SELECT", "SELECT * FROM dbo.users", True),
    (
        "SELECT with WHERE",
        "SELECT * FROM dbo.users WHERE user_type = 'rider'",
        True,
    ),
    (
        "Keyword inside string",
        "SELECT 'DELETE' AS message FROM dbo.users",
        True,
    ),

    # Unsafe queries
    ("DELETE", "DELETE FROM dbo.users", False),
    ("UPDATE", "UPDATE dbo.users SET is_active = 0", False),
    (
        "Multiple statements",
        "SELECT * FROM dbo.users; DELETE FROM dbo.users",
        False,
    ),
    (
        "SELECT INTO",
        "SELECT * INTO dbo.backup_users FROM dbo.users",
        False,
    ),
    (
        "SQL comment",
        "SELECT * FROM dbo.users -- comment",
        False,
    ),
    (
        "Stored procedure",
        "EXEC sp_help",
        False,
    ),
]


print("\n========== SQL SAFETY TESTS ==========\n")

passed = 0

for name, query, expected in test_cases:
    actual, reason = validate_read_only_sql(query)

    success = actual == expected

    if success:
        passed += 1

    print(f"{'PASS' if success else 'FAIL'} | {name}")
    print(f"Expected: {expected} | Actual: {actual}")
    print(f"Reason: {reason}\n")

print(f"Result: {passed}/{len(test_cases)} tests passed.")

if passed == len(test_cases):
    print("All SQL safety regression tests passed.")
else:
    print("Some tests failed. Review the output.")