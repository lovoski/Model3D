#include "BaseModel.h"
#include "Parameters.h"
#include <float.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <CGAL/IO/OBJ.h>
#include <CGAL/Exact_integer.h>
#include <CGAL/Homogeneous.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Surface_mesh/Properties.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include "UnionFind.hpp"
#include "CDT.h"
using namespace std;

namespace Model3D
{
	//////////////////////////////////////////////////////////////////////
	// Construction/Destruction
	//////////////////////////////////////////////////////////////////////
	CBaseModel::CBaseModel(const string& filename) : m_filename(filename), 
		m_Verts(), m_Faces()
	{
	}

	CBaseModel::CBaseModel() : m_Verts(), m_Faces()
	{
	}

	CBaseModel::CBaseModel(const vector<CPoint3D> & vertList, const vector<CBaseModel::CFace>& faceList)
	{
		m_NormalsToVerts.clear();
		m_UselessFaces.clear();

		m_Verts = vertList;
		m_Faces = faceList;
		PreprocessVertsAndFacesIntoBaseModel();
	}

	CBaseModel::CBaseModel(const vector<CPoint3D> & vertList)
	{
		m_NormalsToVerts.clear();
		m_UselessFaces.clear();

		m_Verts = vertList;
		PreprocessVertsAndFacesIntoBaseModel();
	}

	void CBaseModel::PreprocessVertsAndFacesIntoBaseModel()
	{
		m_UselessFaces.clear();
		if (m_Verts.empty())
			return;
		m_NormalsToVerts.resize(m_Verts.size(), CPoint3D(0, 0, 0));
		CPoint3D center(0, 0, 0);
		double sumArea(0);
		CPoint3D sumNormal(0, 0, 0);
		double deta(0);
		for (int i = 0; i < (int)m_Faces.size(); ++i)
		{
			CPoint3D normal = VectorCross(Vert(Face(i)[0]),
				Vert(Face(i)[1]),
				Vert(Face(i)[2]));
			double area = normal.Len();
			CPoint3D gravity3 = Vert(Face(i)[0]) + Vert(Face(i)[1]) + Vert(Face(i)[2]);
			center += area * gravity3;
			sumArea += area;
			sumNormal += normal;
			deta += gravity3 ^ normal;
			normal.x /= area;
			normal.y /= area;
			normal.z /= area;
			for (int j = 0; j < 3; ++j)
			{
				m_NormalsToVerts[Face(i)[j]] += normal;
			}
		}
		center /= sumArea * 3;
		deta -= 3 * (center ^ sumNormal);
		if (true)//deta > 0)
		{
			for (int i = 0; i < GetNumOfVerts(); ++i)
			{
				if (fabs(m_NormalsToVerts[i].x)
					+ fabs(m_NormalsToVerts[i].y)
					+ fabs(m_NormalsToVerts[i].z) >= FLT_EPSILON)
				{
					m_NormalsToVerts[i].Normalize();
				}
			}
		}
		else
		{
			for (int i = 0; i < GetNumOfFaces(); ++i)
			{
				int temp = m_Faces[i][0];
				m_Faces[i][0] = m_Faces[i][1];
				m_Faces[i][1] = temp;
			}
			for (int i = 0; i < GetNumOfVerts(); ++i)
			{
				if (fabs(m_NormalsToVerts[i].x)
					+ fabs(m_NormalsToVerts[i].y)
					+ fabs(m_NormalsToVerts[i].z) >= FLT_EPSILON)
				{
					double len = m_NormalsToVerts[i].Len();
					m_NormalsToVerts[i].x /= -len;
					m_NormalsToVerts[i].y /= -len;
					m_NormalsToVerts[i].z /= -len;
				}
			}
		}

		CPoint3D ptUp(m_Verts[0]);
		CPoint3D ptDown(m_Verts[0]);
		for (int i = 1; i < GetNumOfVerts(); ++i)
		{
			if (m_Verts[i].x > ptUp.x)
				ptUp.x = m_Verts[i].x;
			else if (m_Verts[i].x < ptDown.x)
				ptDown.x = m_Verts[i].x;
			if (m_Verts[i].y > ptUp.y)
				ptUp.y = m_Verts[i].y;
			else if (m_Verts[i].y < ptDown.y)
				ptDown.y = m_Verts[i].y;
			if (m_Verts[i].z > ptUp.z)
				ptUp.z = m_Verts[i].z;
			else if (m_Verts[i].z < ptDown.z)
				ptDown.z = m_Verts[i].z;
		}

		double maxEdgeLenOfBoundingBox = -1;
		if (ptUp.x - ptDown.x > maxEdgeLenOfBoundingBox)
			maxEdgeLenOfBoundingBox = ptUp.x - ptDown.x;
		if (ptUp.y - ptDown.y > maxEdgeLenOfBoundingBox)
			maxEdgeLenOfBoundingBox = ptUp.y - ptDown.y;
		if (ptUp.z - ptDown.z > maxEdgeLenOfBoundingBox)
			maxEdgeLenOfBoundingBox = ptUp.z - ptDown.z;
		m_scale = maxEdgeLenOfBoundingBox / 2;
		//m_center = center;
		//m_ptUp = ptUp;
		//m_ptDown = ptDown;
	}
	double CBaseModel::SignedVolume() const
	{
		double volume(0);
		for (int i = 0; i < m_Faces.size(); ++i)
		{
			Eigen::Matrix4d m;
			m << m_Verts[m_Faces[i][0]].x, m_Verts[m_Faces[i][0]].y, m_Verts[m_Faces[i][0]].z, 1,
				m_Verts[m_Faces[i][1]].x, m_Verts[m_Faces[i][1]].y, m_Verts[m_Faces[i][1]].z, 1,
				m_Verts[m_Faces[i][2]].x, m_Verts[m_Faces[i][2]].y, m_Verts[m_Faces[i][2]].z, 1,
				0, 0, 0, 1;
			volume += 1.0 / 6.0 * m.determinant();
		}
		return volume;
	}

	void CBaseModel::Flip()
	{
		for (int i = 0; i < m_Faces.size(); ++i)
		{
			swap(m_Faces[i][0], m_Faces[i][1]);
		}
	}
	void CBaseModel::LoadModel()
	{
		ReadFile(m_filename);
		PreprocessVertsAndFacesIntoBaseModel();
	}

	string CBaseModel::GetFileShortName() const
	{
		int pos = (int)m_filename.size() - 1;
		while (pos >= 0)
		{
			if (m_filename[pos] == '\\')
				break;
			--pos;
		}
		++pos;
		string str(m_filename.substr(pos));
		return str;
	}

	string CBaseModel::GetFileFullName() const
	{
		return m_filename;
	}

	void CBaseModel::ReadObjFile(const string& filename)
	{
		ifstream in(filename.c_str());
		if (in.fail())
		{
			throw "fail to read file";
		}
		char buf[256];
		while (in.getline(buf, sizeof buf))
		{
			istringstream line(buf);
			string word;
			line >> word;
			if (word == "v")
			{
				CPoint3D pt;
				line >> pt.x;
				line >> pt.y;
				line >> pt.z;

				m_Verts.push_back(pt);
			}
			else if (word == "f")
			{
				CFace face;
				int tmp;
				vector<int> polygon;
				polygon.reserve(4);
				while (line >> tmp)
				{
					polygon.push_back(tmp);
					char tmpBuf[256];
					line.getline(tmpBuf, sizeof tmpBuf, ' ');
				}
				for (int j = 1; j < (int)polygon.size() - 1; ++j)
				{
					face[0] = polygon[0] - 1;
					face[1] = polygon[j] - 1;
					face[2] = polygon[j + 1] - 1;
					m_Faces.push_back(face);
				}
			}
			else
			{
				continue;
			}
		}

		in.close();
	}

	void CBaseModel::ReadFile(const string& filename)
	{
		int nDot = (int)filename.rfind('.');
		if (nDot == -1)
		{
			throw "File name doesn't contain a dot!";
		}
		string extension = filename.substr(nDot + 1);

		if (extension == "obj")
		{
			ReadObjFile(filename);
		}
		else if (extension == "off")
		{
			ReadOffFile(filename);
		}
		else if (extension == "m")
		{
			ReadMFile(filename);
		}
		else
		{
			throw "This format can't be handled!";
		}
	}

	void CBaseModel::ReadString(const string& off_string)
	{
		std::stringstream in;
		in << off_string;
		
		char buf[256];
		in.getline(buf, sizeof buf);
		int vertNum, faceNum, edgeNum;
		in >> vertNum >> faceNum >> edgeNum;

		for (int i = 0; i < vertNum; ++i)
		{
			CPoint3D pt;
			in >> pt.x;
			in >> pt.y;
			in >> pt.z;
			m_Verts.push_back(pt);
		}

		int degree;
		while (in >> degree)
		{
			int first, second;
			in >> first >> second;

			for (int i = 0; i < degree - 2; ++i)
			{
				CFace f;
				f[0] = first;
				f[1] = second;
				in >> f[2];
				m_Faces.push_back(f);
				second = f[2];
			}
		}
		PreprocessVertsAndFacesIntoBaseModel();
	}

	void CBaseModel::ReadOffFile(const string& filename)
	{
		ifstream in(filename.c_str());
		if (in.fail())
		{
			throw "fail to read file";
		}
		char buf[256];
		in.getline(buf, sizeof buf);
		int vertNum, faceNum, edgeNum;
		in >> vertNum >> faceNum >> edgeNum;

		for (int i = 0; i < vertNum; ++i)
		{
			CPoint3D pt;
			in >> pt.x;
			in >> pt.y;
			in >> pt.z;
			m_Verts.push_back(pt);
		}

		int degree;
		while (in >> degree)
		{
			int first, second;
			in >> first >> second;

			for (int i = 0; i < degree - 2; ++i)
			{
				CFace f;
				f[0] = first;
				f[1] = second;
				in >> f[2];
				m_Faces.push_back(f);
				second = f[2];
			}
		}

		in.close();
	}

	void CBaseModel::ReadMFile(const string& filename)
	{
		ifstream in(filename.c_str());
		if (in.fail())
		{
			throw "fail to read file";
		}
		char buf[256];
		while (in.getline(buf, sizeof buf))
		{
			istringstream line(buf);
			if (buf[0] == '#')
				continue;
			string word;
			line >> word;
			if (word == "Vertex")
			{
				int tmp;
				line >> tmp;
				CPoint3D pt;
				line >> pt.x;
				line >> pt.y;
				line >> pt.z;

				m_Verts.push_back(pt);
			}
			else if (word == "Face")
			{
				CFace face;
				int tmp;
				line >> tmp;
				vector<int> polygon;
				polygon.reserve(4);
				while (line >> tmp)
				{
					polygon.push_back(tmp);
				}
				for (int j = 1; j < (int)polygon.size() - 1; ++j)
				{
					face[0] = polygon[0] - 1;
					face[1] = polygon[j] - 1;
					face[2] = polygon[j + 1] - 1;
					m_Faces.push_back(face);
				}
			}
			else
			{
				continue;
			}
		}

		in.close();
	}


	void CBaseModel::SaveMFile(const string& filename) const
	{
		ofstream outFile(filename.c_str());
		for (int i = 0; i < (int)GetNumOfVerts(); ++i)
		{
			outFile << "Vertex " << i + 1 << " " << Vert(i).x << " " << Vert(i).y << " " << Vert(i).z << endl;
		}
		int cnt(0);
		for (int i = 0; i < (int)GetNumOfFaces(); ++i)
		{
			if (m_UselessFaces.find(i) != m_UselessFaces.end())
				continue;
			outFile << "Face " << ++cnt << " " << Face(i)[0] + 1 << " " << Face(i)[1] + 1 << " " << Face(i)[2] + 1 << endl;
		}
		outFile.close();
	}

	void CBaseModel::SaveOffFile(const string& filename) const
	{
		ofstream outFile(filename.c_str());
		outFile << "OFF" << endl;
		outFile << GetNumOfVerts() << " " << GetNumOfFaces() << " " << 0 << endl;
		for (int i = 0; i < (int)GetNumOfVerts(); ++i)
		{
			outFile << Vert(i).x << " " << Vert(i).y << " " << Vert(i).z << endl;
		}
		for (int i = 0; i < (int)GetNumOfFaces(); ++i)
		{
			if (m_UselessFaces.find(i) != m_UselessFaces.end())
				continue;
			outFile << 3 << " " << Face(i)[0] << " " << Face(i)[1] << " " << Face(i)[2] << endl;
		}
		outFile.close();
	}

	void CBaseModel::SaveObjFile(const string& filename) const
	{
		ofstream outFile(filename.c_str());
		outFile << "g " << filename.substr(filename.rfind("\\") + 1, filename.rfind('.') - filename.rfind("\\") - 1) << endl;
		for (int i = 0; i < (int)GetNumOfVerts(); ++i)
		{
			outFile << "v " << std::setiosflags(ios::fixed) << std::setprecision(20) << Vert(i).x << " " << Vert(i).y << " " << Vert(i).z << endl;
		}
		for (int i = 0; i < (int)GetNumOfFaces(); ++i)
		{
			if (m_UselessFaces.find(i) != m_UselessFaces.end())
				continue;
			outFile << "f " << Face(i)[0] + 1 << " " << Face(i)[1] + 1 << " " << Face(i)[2] + 1 << endl;
		}
		outFile.close();
	}

	void CBaseModel::SaveScalarFieldObjFile(const Eigen::VectorXd& vals, const string& filename) const
	{
		ofstream outFile(filename.c_str());
		outFile << "g " << filename.substr(filename.rfind("\\") + 1, filename.rfind('.') - filename.rfind("\\") - 1) << endl;
		outFile << "# maxDis: " << vals.maxCoeff() << endl;
		outFile << "mtllib defaultmaterial.mtl\n"
			<< "usemtl mydefault\n";
		for (int i = 0; i < (int)GetNumOfVerts(); ++i)
		{
			outFile << "v " << std::setiosflags(ios::fixed) << std::setprecision(20) << Vert(i).x << " " << Vert(i).y << " " << Vert(i).z << endl;
		}
		for (int i = 0; i < (int)vals.size(); ++i)
		{
			outFile << "vt " << std::setiosflags(ios::fixed) << std::setprecision(20) << vals(i) << " " << 0 << endl;
		}
		for (int i = 0; i < (int)GetNumOfFaces(); ++i)
		{
			if (m_UselessFaces.find(i) != m_UselessFaces.end())
				continue;
			outFile << "f " << Face(i)[0] + 1 << "/" << Face(i)[0] + 1
				<< " " << Face(i)[1] + 1 << "/" << Face(i)[1] + 1
				<< " " << Face(i)[2] + 1 << "/" << Face(i)[2] + 1 << endl;
		}
		outFile.close();
	}

	void CBaseModel::SaveScalarFieldObjFile(const vector<double>& vals, const string& filename) const
	{
		ofstream outFile(filename.c_str());
		outFile << "g " << filename.substr(filename.rfind("\\") + 1, filename.rfind('.') - filename.rfind("\\") - 1) << endl;
		outFile << "# maxDis: " << *max_element(vals.begin(), vals.end()) << endl;
		outFile << "mtllib defaultmaterial.mtl\n"
			<< "usemtl mydefault\n";
		for (int i = 0; i < (int)GetNumOfVerts(); ++i)
		{
			outFile << "v " << std::setiosflags(ios::fixed) << std::setprecision(20) << Vert(i).x << " " << Vert(i).y << " " << Vert(i).z << endl;
		}
		for (int i = 0; i < (int)vals.size(); ++i)
		{
			outFile << "vt " << std::setiosflags(ios::fixed) << std::setprecision(20) << vals[i] << " " << 0 << endl;
		}
		for (int i = 0; i < (int)GetNumOfFaces(); ++i)
		{
			if (m_UselessFaces.find(i) != m_UselessFaces.end())
				continue;
			outFile << "f " << Face(i)[0] + 1 << "/" << Face(i)[0] + 1
				<< " " << Face(i)[1] + 1 << "/" << Face(i)[1] + 1
				<< " " << Face(i)[2] + 1 << "/" << Face(i)[2] + 1 << endl;
		}
		outFile.close();
	}
	//
	//void CBaseModel::SaveScalarFieldObjFile(const vector<double>& vals, const string& filename) const
	//{
	//	ofstream outFile(filename.c_str());
	//	outFile << "g " << filename.substr(filename.rfind("\\") + 1, filename.rfind('.') - filename.rfind("\\") - 1) << endl;
	//	outFile << "# maxDis: " << *max_element(vals.begin(), vals.end()) << endl;	
	//	for (int i = 0; i < (int)GetNumOfVerts(); ++i)
	//	{
	//		outFile << "v " << std::setiosflags(ios::fixed) << std::setprecision(20) << Vert(i).x << " " << Vert(i).y << " " << Vert(i).z << endl;
	//	}
	//	for (int i = 0; i < (int)vals.size(); ++i)
	//	{
	//		outFile << "vt " << std::setiosflags(ios::fixed) << std::setprecision(20) << vals[i] << " " << 0 << endl;
	//	}
	//	for (int i = 0; i < (int)GetNumOfFaces(); ++i)
	//	{
	//		if (m_UselessFaces.find(i) != m_UselessFaces.end())
	//			continue;
	//		outFile << "f " << Face(i)[0] + 1 << "/" << Face(i)[0] + 1
	//			<< " " << Face(i)[1] + 1 << "/" << Face(i)[1] + 1
	//			<< " " << Face(i)[2] + 1 << "/" << Face(i)[2] + 1 << endl;
	//	}
	//	outFile.close();
	//}
	void CBaseModel::SaveScalarFieldObjFile(const Eigen::VectorXd& vals, double maxV, const string& filename) const
	{
		ofstream outFile(filename.c_str());
		outFile << "g " << filename.substr(filename.rfind("\\") + 1, filename.rfind('.') - filename.rfind("\\") - 1) << endl;
		outFile << "# maxDis = " << vals.maxCoeff() / maxV << endl;
		outFile << "mtllib defaultmaterial.mtl\n"
			<< "usemtl mydefault\n";

		for (int i = 0; i < (int)GetNumOfVerts(); ++i)
		{
			outFile << "v " << std::setiosflags(ios::fixed) << std::setprecision(20) << Vert(i).x << " " << Vert(i).y << " " << Vert(i).z << endl;
		}
		for (int i = 0; i < (int)vals.size(); ++i)
		{
			outFile << "vt " << std::setiosflags(ios::fixed) << std::setprecision(20) << vals(i) / maxV << " " << 0 << endl;
		}
		for (int i = 0; i < (int)GetNumOfFaces(); ++i)
		{
			if (m_UselessFaces.find(i) != m_UselessFaces.end())
				continue;
			outFile << "f " << Face(i)[0] + 1 << "/" << Face(i)[0] + 1
				<< " " << Face(i)[1] + 1 << "/" << Face(i)[1] + 1
				<< " " << Face(i)[2] + 1 << "/" << Face(i)[2] + 1 << endl;
		}
		outFile.close();
	}
	void CBaseModel::SaveScalarFieldObjFile(const vector<double>& vals, double maxV, const string& filename) const
	{
		ofstream outFile(filename.c_str());
		outFile << "g " << filename.substr(filename.rfind("\\") + 1, filename.rfind('.') - filename.rfind("\\") - 1) << endl;
		outFile << "mtllib defaultmaterial.mtl\n"
			<< "usemtl mydefault\n";

		for (int i = 0; i < (int)GetNumOfVerts(); ++i)
		{
			outFile << "v " << std::setiosflags(ios::fixed) << std::setprecision(20) << Vert(i).x << " " << Vert(i).y << " " << Vert(i).z << endl;
		}
		for (int i = 0; i < (int)vals.size(); ++i)
		{
			outFile << "vt " << std::setiosflags(ios::fixed) << std::setprecision(20) << vals[i] / maxV << " " << 0 << endl;
		}
		for (int i = 0; i < (int)GetNumOfFaces(); ++i)
		{
			if (m_UselessFaces.find(i) != m_UselessFaces.end())
				continue;
			outFile << "f " << Face(i)[0] + 1 << "/" << Face(i)[0] + 1
				<< " " << Face(i)[1] + 1 << "/" << Face(i)[1] + 1
				<< " " << Face(i)[2] + 1 << "/" << Face(i)[2] + 1 << endl;
		}
		outFile.close();
	}

	void CBaseModel::SavePamametrizationObjFile(const vector<pair<double, double> >& uvs, const string& filename) const
	{
		ofstream outFile(filename.c_str());
		outFile << "g " << filename.substr(filename.rfind("\\") + 1, filename.rfind('.') - filename.rfind("\\") - 1) << endl;
		outFile << "mtllib defaultmaterial.mtl\n"
			<< "usemtl mydefault\n"; 
		for (int i = 0; i < (int)GetNumOfVerts(); ++i)
		{
			outFile << "v " << Vert(i).x << " " << Vert(i).y << " " << Vert(i).z << endl;
		}
		for (int i = 0; i < (int)uvs.size(); ++i)
		{
			outFile << "vt " << uvs[i].first << " " << uvs[i].second << endl;
		}
		for (int i = 0; i < (int)GetNumOfFaces(); ++i)
		{
			if (m_UselessFaces.find(i) != m_UselessFaces.end())
				continue;
			outFile << "f " << Face(i)[0] + 1 << "/" << Face(i)[0] + 1
				<< " " << Face(i)[1] + 1 << "/" << Face(i)[1] + 1
				<< " " << Face(i)[2] + 1 << "/" << Face(i)[2] + 1 << endl;
		}
		outFile.close();
	}

	void CBaseModel::PrintInfo(ostream& out) const
	{
		out << "Model info is as follows.\n";
		out << "Name: " << GetFileShortName() << endl;
		out << "VertNum = " << GetNumOfVerts() << endl;
		out << "FaceNum = " << GetNumOfFaces() << endl;
		out << "Scale = " << m_scale << endl;
	}

	CPoint3D CBaseModel::GetShiftVertex(int indexOfVert) const
	{
		return Vert(indexOfVert) + Normal(indexOfVert) * RateOfNormalShift * GetScale();
	}

	//CPoint3D CBaseModel::ShiftVertex(int indexOfVert, double epsilon) const
	//{
	//	return Vert(indexOfVert) +  Normal(indexOfVert) * epsilon;
	//}

	int CBaseModel::GetVertexID(const CPoint3D& pt) const
	{
		double dis = DBL_MAX;
		int id;
		for (int i = 0; i < GetNumOfVerts(); ++i)
		{
			if ((Vert(i) - pt).Len() < dis)
			{
				id = i;
				dis = (Vert(id) - pt).Len();
			}
		}
		return id;
	}

	string CBaseModel::GetComments(const char* filename)
	{
		ifstream in(filename);
		char buf[256];
		string result;
		while (in.getline(buf, sizeof buf))
		{
			if (buf[0] == '#')
			{
				result += buf;
				result += "\n";
			}
		}
		in.close();
		return result;
	}

	vector<double> CBaseModel::GetScalarField(string filename)
	{
		vector<double> scalarField;
		ifstream in(filename);
		char buf[256];
		while (in.getline(buf, sizeof buf))
		{
			istringstream line(buf);
			string word;
			line >> word;
			if (word == "vt")
			{
				double value;
				line >> value;
				scalarField.push_back(value);
			}
		}

		in.close();
		return scalarField;
	}

	void CBaseModel::SavePathToObj(const vector<CPoint3D>& pl, const string& filename)
	{
#if 0
		ofstream out(filename.c_str());
		out << "g 3D_Curve" << endl;
		if (!pl.empty())
		{
			for (int i = 0; i < pl.size(); ++i)
			{
				CPoint3D pt = pl[i];
				out << "v " << pt.x << " " << pt.y << " " << pt.z << endl;
			}

			out << "l ";
			for (int i = 0; i < pl.size(); ++i)
			{
				out << i + 1 << " ";
			}
			out << endl;
		}
		out.close();
#else
		vector<pair<CPoint3D, CPoint3D>> pl_new;
		for (int i = 0; i < (int)pl.size() - 1; ++i)
		{
			pl_new.push_back(make_pair(pl[i], pl[i + 1]));
		}
		SaveSegmentsToObj(pl_new, filename);
#endif
	}

	void CBaseModel::SaveSegmentsToObj(const vector<pair<CPoint3D, CPoint3D>>& segs, const string& filename)
	{
		ofstream out(filename.c_str());
		out << "g " << filename.substr(filename.rfind("\\") + 1, filename.rfind('.') - filename.rfind("\\") - 1) << endl;
		int id(0);
		for (auto seg : segs)
		{
			out << "v " << seg.first.x << " " << seg.first.y << " " << seg.first.z << endl;
			out << "v " << seg.second.x << " " << seg.second.y << " " << seg.second.z << endl;
			out << "l " << id + 1 << " " << id + 2 << endl;
			id += 2;
		}
		out.close();
	}

	//void CBaseModel::RemoveUnreferencedVertices()
	//{
	//	set<int> usedIndices;
	//	for (int i = 0; i < GetNumOfFaces(); ++i)
	//	{
	//		if (m_UselessFaces.find(i) != m_UselessFaces.end())
	//			continue;
	//		for (int j = 0; j < 3; ++j)
	//			usedIndices.insert(Face(i)[j]);
	//	}
	//	map<int, int> old2new;
	//	vector<CPoint3D> vertList;
	//	for (int i = 0; i < GetNumOfVerts(); ++i)
	//	{
	//		int sz = old2new.size();
	//		if (usedIndices.find(i) == usedIndices.end())
	//			continue;
	//		old2new[i] = sz;
	//		vertList.push_back(Vert(i));
	//	}
	//	vector<CFace> faceList;
	//	for (int i = 0; i < GetNumOfFaces(); ++i)
	//	{
	//		auto f = Face(i);
	//		for (int j = 0; j < 3; ++j)
	//		{
	//			f[j] = old2new[f[j]];
	//		}
	//		faceList.push_back(f);
	//	}

	//	m_NormalsToVerts.clear();
	//	m_UselessFaces.clear();

	//	swap(m_Verts, vertList);
	//	swap(m_Faces, faceList);
	//	PreprocessVertsAndFacesIntoBaseModel();
	//}
	map<CPoint3D, int> CBaseModel::AddFaceInteriorPoints(const vector<pair<int, CPoint3D>> &pts)
	{
		map<CPoint3D, int> idsOfInsertedPoints;
		map<int, set<CPoint3D>> pts_organized_by_faces;
		for (int i = 0; i < pts.size(); ++i)
		{
			pts_organized_by_faces[pts[i].first].insert(pts[i].second);
		}

		for (auto elem_pair : pts_organized_by_faces)
		{
			int faceID = elem_pair.first;
			auto dir1 = Vert(Face(faceID)[1]) - Vert(Face(faceID)[0]);
			dir1.Normalize();
			auto normal = VectorCross(Vert(Face(faceID)[0]),
				Vert(Face(faceID)[1]), Vert(Face(faceID)[2]));
			auto dir2 = normal * dir1;
			dir2.Normalize();
			vector<CDT_Kernel::Point_2> pts;
			map<CDT_Kernel::Point_2, int> pts2ID;
			for (int j = 0; j < 3; ++j)
			{
				pts.push_back(CDT_Kernel::Point_2(dir1 ^ Vert(Face(faceID)[j]), dir2 ^ Vert(Face(faceID)[j])));
				pts2ID[pts.back()] = Face(faceID)[j];
			}
			for (auto pt3d : elem_pair.second)
			{
				pts.push_back(CDT_Kernel::Point_2(dir1 ^ pt3d, dir2 ^ pt3d));
				pts2ID[pts.back()] = m_Verts.size();
				idsOfInsertedPoints[pt3d] = m_Verts.size();
				m_Verts.push_back(pt3d);
			}
			CDT cdt;
			cdt.insert_constraint(pts.begin(), pts.begin() + 3, true);
			cdt.insert(pts.begin() + 3, pts.end());
			mark_domains(cdt);
			for (CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
				fit != cdt.finite_faces_end(); ++fit)
			{
				if (fit->info().in_domain())
				{
					int id1 = pts2ID[fit->vertex(0)->point()];
					int id2 = pts2ID[fit->vertex(1)->point()];
					int id3 = pts2ID[fit->vertex(2)->point()];
					//cerr << "ids : " << id1 << ", " << id2 << ", " << id3 << endl;
					m_Faces.push_back(CFace(id1, id2, id3));
				}
			}
			m_UselessFaces.insert(faceID);
		}
		vector<CFace> faceList;
		for (int i = 0; i < GetNumOfFaces(); ++i)
		{
			if (m_UselessFaces.find(i) == m_UselessFaces.end())
				faceList.push_back(Face(i));
		}
		swap(faceList, m_Faces);
		PreprocessVertsAndFacesIntoBaseModel();
		return idsOfInsertedPoints;
	}
	int CBaseModel::AddFaceInteriorPoints(int faceIndex, CPoint3D pt)
	{
		vector<pair<int, CPoint3D>> pts;
		pts.push_back(make_pair(faceIndex, pt));
		auto res = AddFaceInteriorPoints(pts);
		return res.begin()->second;
	}

	vector<CBaseModel::CFace> CBaseModel::GetUsefulFaces() const
	{
		vector<CFace> faceList;
		for (int i = 0; i < m_Faces.size(); ++i)
		{
			if (m_UselessFaces.find(i) != m_UselessFaces.end())
				continue;
			faceList.push_back(m_Faces[i]);
		}
		return faceList;
	}
	//void CBaseModel::RemoveShortEdges(double coe) //less than 
	//{
	//	double length_sum(0);
	//	int cnt(0);
	//	for (int i = 0; i < m_Faces.size(); ++i)
	//	{
	//		for (int j = 0; j < 3; ++j)
	//		{
	//			int nxt = (j + 1) % 3;
	//			double len = (Vert(Face(i)[j]) - Vert(Face(i)[nxt])).Len();
	//			length_sum += len;
	//			cnt++;
	//		}
	//	}
	//	double average_len = length_sum / cnt;
	//	UnionFind uf_vertices(m_Verts.size());
	//	for (int i = 0; i < m_Faces.size(); ++i)
	//	{
	//		for (int j = 0; j < 3; ++j)
	//		{
	//			int nxt = (j + 1) % 3;
	//			double len = (Vert(Face(i)[j]) - Vert(Face(i)[nxt])).Len();
	//			if (len < coe * average_len)
	//			{
	//				uf_vertices.AddConnection(Face(i)[j], Face(i)[nxt]);
	//			}
	//		}
	//	}
	//	uf_vertices.UpdateParents2Ancestors();
	//	vector<CFace> new_faces;
	//	UnionFind uf_faces(m_Faces.size());
	//	map<pair<int, int>, set<int>> facesWithEdge;
	//	for (int i = 0; i < m_Faces.size(); ++i)
	//	{
	//		set<int> ids;
	//		for (int j = 0; j < 3; ++j)
	//		{
	//			ids.insert(uf_vertices.FindAncestor(Face(i)[j]));
	//		}
	//		if (ids.size() < 3)
	//			continue;
	//		new_faces.push_back(CFace(uf_vertices.FindAncestor(Face(i)[0]),
	//			uf_vertices.FindAncestor(Face(i)[1]), uf_vertices.FindAncestor(Face(i)[2])));
	//		facesWithEdge[make_pair(min(new_faces.back()[0], new_faces.back()[1]),
	//			max(new_faces.back()[0], new_faces.back()[1]))].insert(new_faces.size() - 1);
	//		facesWithEdge[make_pair(min(new_faces.back()[1], new_faces.back()[2]),
	//			max(new_faces.back()[1], new_faces.back()[2]))].insert(new_faces.size() - 1);
	//		facesWithEdge[make_pair(min(new_faces.back()[2], new_faces.back()[0]),
	//			max(new_faces.back()[2], new_faces.back()[0]))].insert(new_faces.size() - 1);
	//	}		
	//	for (auto mypair : facesWithEdge)
	//	{
	//		if (mypair.second.size() != 2)
	//			continue;
	//		uf_faces.AddConnection(*mypair.second.begin(), *mypair.second.rbegin());
	//	}

	//	m_Faces.clear();
	//	for (auto mypair : uf_faces.GetNontrivialClusters())
	//	{
	//		if (mypair.second.size() >= 4)
	//		{
	//			for (int k = 0; k<mypair.second.size(); ++k)
	//				m_Faces.push_back(new_faces[mypair.second[k]]);
	//		}
	//	}

	//	for (auto mypair : uf_vertices.GetNontrivialClusters())
	//	{
	//		CPoint3D ptSum(CPoint3D::Origin());
	//		for (int i = 0; i < mypair.second.size(); ++i)
	//		{
	//			ptSum = ptSum + Vert(mypair.second[i]);
	//		}
	//		m_Verts[mypair.first] = ptSum / mypair.second.size();
	//	}

	//	RemoveUnreferencedVertices();
	//}
}

