import React, { useEffect, useState } from 'react';
import { Row, Collapse, Col, Table, Checkbox } from 'antd';
import { useModel } from 'umi';
import { IMenuRecord } from '@/models/menus';

const TableMenu = () => {
  const groupmenusModel = useModel('groupmenu');
  useEffect(() => {
    groupmenusModel.getData().then(() => {
      // console.log(groupmenusModel.menusList, 'groupmenusModel.menusList');
      // console.log(groupmenusModel.groupsList, 'groupmenusModel.groupsList');
    });
  }, []);
  return (
    <React.Fragment>
      {groupmenusModel.menusList.length > 0 && groupmenusModel.groupsList.length > 0 && (
        <Table
          bordered
          title={() => <h3>Phân quyền nhóm người dùng</h3>}
          columns={[
            {
              title: 'STT',
              dataIndex: 'index',
              width: 80,
              align: 'center',
            },
            {
              title: 'Nhóm người dùng',
              dataIndex: 'group_name',
              width: 160,
              align: 'center',
            },
          ].concat(
            groupmenusModel.menusList.map((item: any, index: number) => ({
              title: item.name,
              dataIndex: item.id,
              key: item.id,
              with: 120,
              render: (value: any, record: any) => {
                // console.log(record, 'record');
                return (
                  <Checkbox
                    checked={record[index]}
                    onChange={(e) => {
                      if (e.target.checked) {
                        groupmenusModel.add({
                          group_id: record.id,
                          menu_id: item.id,
                        });
                      } else {
                        groupmenusModel.del({
                          group_id: record.id,
                          menu_id: item.id,
                        });
                      }
                      // window.location.reload();
                    }}
                  />
                );
              },
            })),
          )}
          dataSource={groupmenusModel.groupsList?.map((item: any, index: number) => {
            return {
              ...item?.menu,
              group_name: item?.group?.name,
              id: item?.group?.id,
              index: index + 1,
            };
          }, [])}
          pagination={true}
          scroll={{ x: 1500 }}
          rowKey={(record) => record.id}
        />
      )}
    </React.Fragment>
  );
};

export default TableMenu;
