import React, { useState } from 'react';
import { Users, Plus, DollarSign, Phone, Mail, Star, Building2, Filter, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui';
import { useERP } from '../../hooks/useERP';

interface Customer {
  id: string;
  name: string;
  company: string;
  email: string;
  phone: string;
  status: 'lead' | 'prospect' | 'customer' | 'churned';
  ltv: number;
  lastContact: string;
  rating: number;
}

interface Deal {
  id: string;
  customerName: string;
  title: string;
  value: number;
  stage: 'lead' | 'qualified' | 'proposal' | 'negotiation' | 'closed_won' | 'closed_lost';
  probability: number;
  expectedClose: string;
}

export const CRM: React.FC = () => {
  const { erpData, createCustomer } = useERP();
  const [activeTab, setActiveTab] = useState<'customers' | 'deals' | 'analytics'>('customers');
  const [showNewCustomer, setShowNewCustomer] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [newCustomer, setNewCustomer] = useState({
    name: '',
    company: '',
    email: '',
    phone: '',
    status: 'lead'
  });

  const crmData = erpData as any;
  const stats = [
    { title: 'Total Customers', value: crmData?.totalCustomers || 0, icon: Users, color: 'blue' },
    { title: 'Active Deals', value: crmData?.activeDeals || 0, icon: TrendingUp, color: 'green' },
    { title: 'Pipeline Value', value: `$${(crmData?.pipelineValue || 0).toLocaleString()}`, icon: DollarSign, color: 'purple' },
    { title: 'Avg LTV', value: `$${(crmData?.avgLTV || 0).toLocaleString()}`, icon: Star, color: 'yellow' }
  ];

  const dealStages = [
    { name: 'Lead', count: 12, value: 45000, color: 'bg-gray-500' },
    { name: 'Qualified', count: 8, value: 78000, color: 'bg-blue-500' },
    { name: 'Proposal', count: 5, value: 125000, color: 'bg-yellow-500' },
    { name: 'Negotiation', count: 3, value: 89000, color: 'bg-orange-500' },
    { name: 'Closed Won', count: 15, value: 245000, color: 'bg-green-500' },
    { name: 'Closed Lost', count: 7, value: 95000, color: 'bg-red-500' }
  ];

  const filteredCustomers = crmData?.customers?.filter((customer: Customer) =>
    customer.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    customer.company.toLowerCase().includes(searchTerm.toLowerCase()) ||
    customer.email.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleCreateCustomer = async () => {
    await createCustomer(newCustomer);
    setShowNewCustomer(false);
    setNewCustomer({ name: '', company: '', email: '', phone: '', status: 'lead' });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'lead': return 'bg-gray-100 text-gray-800';
      case 'prospect': return 'bg-blue-100 text-blue-800';
      case 'customer': return 'bg-green-100 text-green-800';
      case 'churned': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStageColor = (stage: string) => {
    switch (stage) {
      case 'lead': return 'bg-gray-100 text-gray-800';
      case 'qualified': return 'bg-blue-100 text-blue-800';
      case 'proposal': return 'bg-yellow-100 text-yellow-800';
      case 'negotiation': return 'bg-orange-100 text-orange-800';
      case 'closed_won': return 'bg-green-100 text-green-800';
      case 'closed_lost': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Customer Relationship Management</h1>
          <p className="text-gray-600 mt-1">Manage customers, deals, and sales pipeline</p>
        </div>
        <button
          onClick={() => setShowNewCustomer(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus size={20} />
          Add Customer
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
          {(['customers', 'deals', 'analytics'] as const).map((tab) => (
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
      {activeTab === 'customers' && (
        <div className="space-y-6">
          {/* Search */}
          <div className="flex gap-4">
            <input
              type="text"
              placeholder="Search customers..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="flex-1 px-4 py-2 border rounded-lg"
            />
            <button className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-gray-50">
              <Filter size={20} />
              Filter
            </button>
          </div>

          {/* Customers Table */}
          <Card>
            <CardHeader>
              <CardTitle>Customer Directory</CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Name</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Company</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Contact</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Status</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">LTV</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Rating</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {filteredCustomers?.map((customer: Customer) => (
                    <tr key={customer.id}>
                      <td className="px-4 py-3 font-medium">{customer.name}</td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <Building2 size={16} className="text-gray-400" />
                          {customer.company}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2 text-sm">
                            <Mail size={14} className="text-gray-400" />
                            {customer.email}
                          </div>
                          <div className="flex items-center gap-2 text-sm">
                            <Phone size={14} className="text-gray-400" />
                            {customer.phone}
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded text-sm ${getStatusColor(customer.status)}`}>
                          {customer.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-medium">${customer.ltv.toLocaleString()}</td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-1">
                          {[...Array(5)].map((_, i) => (
                            <Star
                              key={i}
                              size={16}
                              className={i < customer.rating ? 'text-yellow-400 fill-yellow-400' : 'text-gray-300'}
                            />
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === 'deals' && (
        <div className="space-y-6">
          {/* Pipeline Overview */}
          <Card>
            <CardHeader>
              <CardTitle>Sales Pipeline</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex overflow-x-auto pb-4">
                {dealStages.map((stage) => (
                  <div key={stage.name} className="flex-shrink-0 w-48 mr-4">
                    <div className={`${stage.color} text-white rounded-t-lg p-3`}>
                      <div className="font-medium">{stage.name}</div>
                      <div className="text-sm opacity-90">{stage.count} deals</div>
                    </div>
                    <div className="bg-gray-50 border border-t-0 rounded-b-lg p-3">
                      <div className="text-lg font-bold">${(stage.value / 1000).toFixed(0)}k</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Deals Table */}
          <Card>
            <CardHeader>
              <CardTitle>Active Deals</CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Deal</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Customer</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Value</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Stage</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Probability</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-700">Expected Close</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {crmData?.deals?.map((deal: Deal) => (
                    <tr key={deal.id}>
                      <td className="px-4 py-3 font-medium">{deal.title}</td>
                      <td className="px-4 py-3 text-gray-600">{deal.customerName}</td>
                      <td className="px-4 py-3 font-medium">${deal.value.toLocaleString()}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded text-sm ${getStageColor(deal.stage)}`}>
                          {deal.stage.replace('_', ' ')}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ width: `${deal.probability}%` }}
                            ></div>
                          </div>
                          <span className="text-sm">{deal.probability}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-gray-600">{deal.expectedClose}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === 'analytics' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Conversion Funnel</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { stage: 'Leads', count: 100, percentage: 100 },
                  { stage: 'Qualified', count: 68, percentage: 68 },
                  { stage: 'Proposals', count: 42, percentage: 42 },
                  { stage: 'Negotiations', count: 28, percentage: 28 },
                  { stage: 'Closed Won', count: 15, percentage: 15 }
                ].map((item) => (
                  <div key={item.stage}>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-medium">{item.stage}</span>
                      <span className="text-sm text-gray-600">{item.count} ({item.percentage}%)</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className="bg-blue-500 h-3 rounded-full"
                        style={{ width: `${item.percentage}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Monthly Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center p-4 bg-green-50 rounded-lg">
                  <div>
                    <div className="text-sm text-gray-600">Revenue This Month</div>
                    <div className="text-2xl font-bold text-green-700">$125,000</div>
                  </div>
                  <TrendingUp className="text-green-600" size={32} />
                </div>
                <div className="flex justify-between items-center p-4 bg-blue-50 rounded-lg">
                  <div>
                    <div className="text-sm text-gray-600">New Customers</div>
                    <div className="text-2xl font-bold text-blue-700">24</div>
                  </div>
                  <Users className="text-blue-600" size={32} />
                </div>
                <div className="flex justify-between items-center p-4 bg-purple-50 rounded-lg">
                  <div>
                    <div className="text-sm text-gray-600">Avg Deal Size</div>
                    <div className="text-2xl font-bold text-purple-700">$16,350</div>
                  </div>
                  <DollarSign className="text-purple-600" size={32} />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* New Customer Modal */}
      {showNewCustomer && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-xl font-bold mb-4">Add New Customer</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  value={newCustomer.name}
                  onChange={(e) => setNewCustomer({ ...newCustomer, name: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Company</label>
                <input
                  type="text"
                  value={newCustomer.company}
                  onChange={(e) => setNewCustomer({ ...newCustomer, company: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input
                  type="email"
                  value={newCustomer.email}
                  onChange={(e) => setNewCustomer({ ...newCustomer, email: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                <input
                  type="tel"
                  value={newCustomer.phone}
                  onChange={(e) => setNewCustomer({ ...newCustomer, phone: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
                <select
                  value={newCustomer.status}
                  onChange={(e) => setNewCustomer({ ...newCustomer, status: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg"
                >
                  <option value="lead">Lead</option>
                  <option value="prospect">Prospect</option>
                  <option value="customer">Customer</option>
                </select>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={handleCreateCustomer}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Create
              </button>
              <button
                onClick={() => setShowNewCustomer(false)}
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

export default CRM;
