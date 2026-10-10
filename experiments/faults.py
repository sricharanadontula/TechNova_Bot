from collections import Counter

print("=" * 60)
print("TECHNOVA RESILIENCE TEST - 20 REQUESTS")
print("=" * 60)


# --------------------------------------------------
# ERROR COUNTER
# --------------------------------------------------

error_counter = Counter()


# --------------------------------------------------
# SIMULATED REQUEST HANDLER
# --------------------------------------------------

def process_request(request_id, fault_type=None):

    try:

        # Scenario 1: invalid model
        if fault_type == "invalid_model":
            raise RuntimeError("Model endpoint returned 404")

        # Scenario 2: invalid API key
        elif fault_type == "invalid_api_key":
            raise PermissionError("API key authentication failed")

        # Scenario 3: timeout
        elif fault_type == "timeout":
            raise TimeoutError("Model request timed out")

        # Scenario 4: tool failure
        elif fault_type == "tool_failure":
            raise RuntimeError("Order database connection failed")

        # Scenario 5: bad input
        elif fault_type == "bad_input":
            raise ValueError("Invalid or empty user input")

        # Normal request
        else:
            return True

    except TimeoutError:
        error_counter["timeout"] += 1
        return False

    except PermissionError:
        error_counter["invalid_api_key"] += 1
        return False

    except ValueError:
        error_counter["bad_input"] += 1
        return False

    except RuntimeError:

        if fault_type == "invalid_model":
            error_counter["invalid_model"] += 1

        elif fault_type == "tool_failure":
            error_counter["tool_failure"] += 1

        return False


# --------------------------------------------------
# 20 REQUESTS
# --------------------------------------------------

requests = [
    None,
    None,
    "bad_input",
    None,
    "timeout",
    None,
    None,
    "tool_failure",
    None,
    None,
    "invalid_model",
    None,
    None,
    "bad_input",
    None,
    "timeout",
    None,
    "tool_failure",
    None,
    "invalid_api_key",
]


successful_requests = 0
failed_requests = 0


for i, fault in enumerate(requests, start=1):

    success = process_request(i, fault)

    if success:
        successful_requests += 1
        status = "SUCCESS"
    else:
        failed_requests += 1
        status = "ERROR"

    print(
        f"Request {i:02d} | "
        f"{status}"
        + (f" | {fault}" if fault else "")
    )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

total_requests = len(requests)

error_rate = (
    failed_requests / total_requests
) * 100


print("\n" + "=" * 60)
print("RESILIENCE TEST RESULTS")
print("=" * 60)

print(f"Total requests:      {total_requests}")
print(f"Successful requests: {successful_requests}")
print(f"Failed requests:     {failed_requests}")
print(f"Error count:         {failed_requests}")
print(f"Error rate:          {error_rate:.2f}%")


print("\nErrors by category:")

for error_type, count in error_counter.items():
    print(f"- {error_type}: {count}")


print("\n" + "=" * 60)