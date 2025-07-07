import matplotlib.pyplot as plt
import numpy as np

# Intervallo x (da 1 a 20 o più, ora possiamo spingerci oltre)
x = np.arange(10, 16)
n = 2 ** x


# Dimensioni in byte
sizeof_double = 8
sizeof_int = 4
density = 0.15

# Conversione a GB
to_kb = 1e3

# Memoria in formato COO (in GB)
mem_coo = (((n/to_kb) * (n/to_kb) * density * (sizeof_double + 2 * sizeof_int)) / to_kb )*2 + (n/to_kb)*(n/to_kb)*(sizeof_double/to_kb)

# Memoria in formato CSR (in GB)
mem_csr = ((n/to_kb) * (n/to_kb) * density * (sizeof_double + sizeof_int)/ to_kb + (((n + 1)/ to_kb) * sizeof_int / (to_kb*to_kb)))*2 + (n/to_kb)*(n/to_kb)*(sizeof_double/to_kb)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(n, mem_coo, label='COO', marker='o')
plt.plot(n, mem_csr, label='CSR', marker='s')

plt.xlabel('Square Matrix dimension')
plt.ylabel('Memory (GB)')
plt.title('Memory Usage (COO vs CSR)')
#plt.xticks(x)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
