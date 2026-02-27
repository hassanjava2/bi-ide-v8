import React, { useState } from 'react';
import { Package, Plus, AlertTriangle, Search, Filter, Warehouse } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui';
import { useERP } from '../../hooks/useERP';

interface Product {
  id: string;
  name: string;
  sku: string;
  category: string;
  quantity: number;
  reorderPoint: number;
  unitPrice: number;
  warehouse: string;
  status: 'in_stock' | 'low_stock' | 'out_of_stock';
}

// @ts-ignore
interface StockMovement {
  id: string;
  productName: string;
  type: 'in' | 'out' | 'adjustment';
  quantity: number;
  date: string;
  reason: string;
}

export const Inventory: React.FC = () => {
  const { erpData, createProduct } = useERP();
  const [activeTab, setActiveTab] = useState<'products' | 'movements' | 'warehouses'>('products');
  const [searchTerm, setSearchTerm] = useState('');
  const [showNewProduct, setShowNewProduct] = useState(false);
  const [newProduct, setNewProduct] = useState({
    name: '',
    sku: '',
    category: '',
    quantity: 0,
    reorderPoint: 10,
    unitPrice: 0,
    warehouse: ''
  });

  const inventoryData = erpData;
  const stats = [
    { title: 'Total Products', value: inventoryData?.totalProducts || 0, icon: Package, color: 'blue' },
    { title: 'In Stock', value: inventoryData?.inStock || 0, icon: Warehouse, color: 'green' },
    { title: 'Low Stock', value: inventoryData?.lowStock || 0, icon: AlertTriangle, color: 'yellow' },
    { title: 'Out of Stock', value: inventoryData?.outOfStock || 0, icon: AlertTriangle, color: 'red' }
  ];

  const categories = [
    { name: 'Electronics', count: 45 },
    { name: 'Office Supplies', count: 120 },
    { name: 'Furniture', count: 25 },
    { name: 'Software Licenses', count: 15 },
    { name: 'Services', count: 8 }
  ];

  const filteredProducts = inventoryData?.products?.filter((product: Product) =>
    product.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    product.sku.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleCreateProduct = async () => {
    await createProduct(newProduct);
    setShowNewProduct(false);
    setNewProduct({ name: '', sku: '', category: '', quantity: 0, reorderPoint: 10, unitPrice: 0, warehouse: '' });
  };

  const getStockStatusColor = (status: string) => {
    switch (status) {
      case 'in_stock': return 'bg-green-100 text-green-800';
      case 'low_stock': return 'bg-yellow-100 text-yellow-800';
      case 'out_of_stock': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Inventory Management</h1>
          <p className="text-gray-600 mt-1">Track products, stock levels, and movements</p>
        </div>
        <button
          onClick={() => setShowNewProduct(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus size={20} />
          Add Product
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <Card key={index}>
            <CardContent className="flex items-center p-6">
              <div className={`p-3 rounded-lg bg-${stat.color}-100`}>
                <stat.icon className={`text-${stat.color}-600`} size={24} />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">{stat.title}</p>
                <p className="text-2xl font-bold">{stat.value}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex gap-8">
          {(['products', 'movements', 'warehouses'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-4 px-1 border-b-2 font-medium capitalize ${
                activeTab === tab
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'products' && (
        <div className="space-y-6">
          {/* Search and Filter */}
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
              <input
                type="text"
                placeholder="Search products by name or SKU..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border rounded-lg"
              />
            </div>
            <button className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-gray-50">
              <Filter size={20} />
              Filter
            </button>
          </div>

          {/* Categories */}
          <Card>
            <CardHeader>
              <CardTitle>Categories</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3">
                {categories.map((cat) => (
                  <div key={cat.name} className="px-4 py-2 bg-gray-100 rounded-lg">
                    {cat.name}: {cat.count}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Products Table */}
          <Card>
            <CardHeader>
              <CardTitle>Products</CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Product</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">SKU</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Category</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Stock</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Unit Price</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Status</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Warehouse</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {filteredProducts?.map((product: Product) => (
                    <tr key={product.id}>
                      <td className="px-4 py-3 font-medium">{product.name}</td>
                      <td className="px-4 py-3 text-gray-600">{product.sku}</td>
                      <td className="px-4 py-3">
                        <span className="px-2 py-1 bg-gray-100 rounded text-sm">
                          {product.category}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{product.quantity}</span>
                          {product.quantity <= product.reorderPoint && (
                            <AlertTriangle size={16} className="text-yellow-500" />
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3">${product.unitPrice.toFixed(2)}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded text-sm ${getStockStatusColor(product.status)}`}>
                          {product.status.replace('_', ' ')}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-gray-600">{product.warehouse}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === 'movements' && (
        <Card>
          <CardHeader>
            <CardTitle>Stock Movements</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Product</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Type</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Quantity</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Date</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {(inventoryData as any)?.movements?.map((movement: any) => (
                  <tr key={movement.id}>
                    <td className="px-4 py-3 font-medium">{movement.productName}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-sm ${
                        movement.type === 'in' ? 'bg-green-100 text-green-800' :
                        movement.type === 'out' ? 'bg-red-100 text-red-800' :
                        'bg-yellow-100 text-yellow-800'
                      }`}>
                        {movement.type}
                      </span>
                    </td>
                    <td className={`px-4 py-3 font-medium ${
                      movement.type === 'in' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {movement.type === 'in' ? '+' : '-'}{movement.quantity}
                    </td>
                    <td className="px-4 py-3 text-gray-600">{movement.date}</td>
                    <td className="px-4 py-3 text-gray-600">{movement.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {activeTab === 'warehouses' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {['Main Warehouse', 'East Branch', 'West Branch'].map((warehouse) => (
            <Card key={warehouse}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Warehouse size={20} />
                  {warehouse}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Products</span>
                    <span className="font-medium">45</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total Value</span>
                    <span className="font-medium">$125,000</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Space Used</span>
                    <span className="font-medium">78%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div className="bg-blue-500 h-2 rounded-full" style={{ width: '78%' }}></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* New Product Modal */}
      {showNewProduct && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-xl font-bold mb-4">Add New Product</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  value={newProduct.name}
                  onChange={(e) => setNewProduct({ ...newProduct, name: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">SKU</label>
                <input
                  type="text"
                  value={newProduct.sku}
                  onChange={(e) => setNewProduct({ ...newProduct, sku: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                <select
                  value={newProduct.category}
                  onChange={(e) => setNewProduct({ ...newProduct, category: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                >
                  <option value="">Select Category</option>
                  {categories.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Quantity</label>
                  <input
                    type="number"
                    value={newProduct.quantity}
                    onChange={(e) => setNewProduct({ ...newProduct, quantity: Number(e.target.value) })}
                    className="w-full px-3 py-2 border rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Reorder Point</label>
                  <input
                    type="number"
                    value={newProduct.reorderPoint}
                    onChange={(e) => setNewProduct({ ...newProduct, reorderPoint: Number(e.target.value) })}
                    className="w-full px-3 py-2 border rounded-lg"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Unit Price</label>
                <input
                  type="number"
                  step="0.01"
                  value={newProduct.unitPrice}
                  onChange={(e) => setNewProduct({ ...newProduct, unitPrice: Number(e.target.value) })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Warehouse</label>
                <select
                  value={newProduct.warehouse}
                  onChange={(e) => setNewProduct({ ...newProduct, warehouse: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                >
                  <option value="">Select Warehouse</option>
                  <option value="Main Warehouse">Main Warehouse</option>
                  <option value="East Branch">East Branch</option>
                  <option value="West Branch">West Branch</option>
                </select>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCreateProduct}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Create
              </button>
              <button
                onClick={() => setShowNewProduct(false)}
                className="flex-1 px-4 py-2 border rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Inventory;
